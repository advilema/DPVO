import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .net import VONet
from .utils import *
from . import projective_ops as pops
import torch_tensorrt
import random

import time

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:
    def __init__(self, cfg, network, poses_gt=None, ht=480, wd=640, viz=False, enable_timing=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = enable_timing
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME  # number of patches per frame
        self.N = self.cfg.BUFFER_SIZE  # maximum length of self.poses_, so the maximum number of poses and frames we can save

        self.ht = ht    # visualization image height
        self.wd = wd    # visualization image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        if self.cfg.MOTION_MODEL == 'GT':
            assert poses_gt is not None, "to use MOTION_MODEL GT you need to pass a valid poses_gt matrix"
            poses_gt_tens_list = [SE3(torch.from_numpy(pose).to(device="cuda")) for pose in poses_gt]
            poses_gt_lietorch = lietorch.stack(poses_gt_tens_list, dim=0)
            poses_gt_inv_tens = poses_gt_lietorch.inv().data.float()
            self.poses_gt = poses_gt_inv_tens
        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        # dimensions explanation: self.N: max number of frames, self.M: max number of patches, self.P patches dimension
        # 3: x, y coordinates of patch in frame, and d, inverse depth map of the patch
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32  # probably that's the maximum number of feature maps that we store, analogous to self.N for the frames

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)  # patches in the context features
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)  # patches in the matching features

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)  # finer pyramidal feature map
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)  # coarser pyramidal feature map

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        # patch self.kk[ind] is present in frames self.ii[ind] and self.jj[ind]
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")  # starting frame id
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")  # ending frame id
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")  # patch id

        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None
        if viz:
            self.start_viewer()

        #self.lmbda = torch.as_tensor([1e-4], device="cuda")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network



        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        # dimensios explanation: self.N*self.M tot max number of patches that we can track, self.P patches dimension
        return self.patches_.view(1, self.N*self.M, 3, self.P, self.P)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """

        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)  # patches indices
        jj1 = jj % (self.mem)  # frames indices
        #print(self.gmap.shape)
        # if coords index is out of range it gets reassigned. It could be interesting to remove the matches that are out
        # of the coordinates range
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        # patches shape: [1, 196608, 3, 3, 3]
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        # if not enough motion between frame i and j, remove the frames from the patches graph
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        # remove all frames that are older than cfg.REMOVAL_WINDOW frames ago
        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            # reproject the patches from the ii frames to the jj frames
            coords = self.reproject()  # shape: [batch size X n of matches X 2 (x,y coords) X patch dim X patch dim]
            # checking if the reprojected patch is outside the feature map
            maskxmax = coords.amax(dim=(3, 4))[0, :, 0] > self.pyramid[0].shape[3]
            maskymax = coords.amax(dim=(3, 4))[0, :, 1] > self.pyramid[0].shape[4]
            maskxmin = coords.amin(dim=(3, 4))[0, :, 0] < 0
            maskymin = coords.amin(dim=(3, 4))[0, :, 1] < 0
            maskmax = torch.logical_or(maskxmax, maskymax)
            maskmin = torch.logical_or(maskxmin, maskymin)
            # all of these elements fall outside the feature map
            outliers_mask = torch.logical_or(maskmax, maskmin)
            #print('frame diff:  ', self.ii[outliers_mask] - self.jj[outliers_mask])
            #self.remove_factors(outliers_mask)
            #coords = coords[:, torch.logical_not(outliers_mask), :, :, :]
            #print(self.jj.shape)
            # TODO: Fare un tentativo anche con GT per vedere se funziona meglio
            # TODO: cercare di capire com'è la correlazione, il delta e i weghts nei punti in cui c'è la maschera

            with autocast(enabled=True):
                # compute correlation features between reprojected patches
                corr = self.corr(coords)
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                start_net_update = time.perf_counter()
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, self.ii, self.jj, self.kk)
                stop_net_update = time.perf_counter()


            # TODO: this is taking way too long, make lmbda a state variable
            start_lambda = time.perf_counter()
            lmbda = torch.as_tensor([1e-4], device="cuda")
            stop_lambda = time.perf_counter()

            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()
        with open('times_net_update.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_net_update - start_net_update)]))
        with open('times_lambda.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_lambda - start_lambda)]))

        start_ba = time.perf_counter()
        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            print('shape', weight.shape)
            print('weights', weight)
            print('\n\n\n\n')

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
        stop_ba = time.perf_counter()
        with open('times_ba.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_ba - start_ba)]))
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if self.viewer is not None:
            self.viewer.update_image(image)
            self.viewer.loop()

        image = 2 * (image[None,None] / 255.0) - 0.5

        # patchify the images
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            start = time.perf_counter()
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    gradient_bias=self.cfg.GRADIENT_BIAS, 
                    return_color=True)
            stop = time.perf_counter()
        with open('times_patchify.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop - start)]))

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            # TODO: usare Kalman filter qua
            # estimate the next pose assuming a linear motion model
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            # use the previous pose as an estimate for the next pose
            elif self.cfg.MOTION_MODEL == 'GT':
                previous_gt_pose = SE3(self.poses_gt[self.counter - 1])
                current_gt_pose = SE3(self.poses_gt[self.counter])
                current_pose = SE3(self.poses_[self.n-1])
                pose_update = current_gt_pose * previous_gt_pose.inv()
                tvec_qvec = (pose_update * current_pose).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses_[self.n - 1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            self.update()
            start = time.perf_counter()
            self.keyframe()
            stop = time.perf_counter()
            with open('times_keyframe.txt', 'a+') as file:
                np.savetxt(file, np.array([1000*(stop - start)]))
