import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3
import time

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, ii, jj, kk):
        """ update operator """

        # correlation
        start_netcorr = time.perf_counter()
        net = net + inp + self.corr(corr)
        net = self.norm(net)
        stop_netcorr = time.perf_counter()
        with open('times_corr.txt', 'a+') as file:
            np.savetxt(file, np.array([1000 * (stop_netcorr - start_netcorr)]))

        # Find neighbours for the convolution
        start_conv = time.perf_counter()
        ix, jx = fastba.neighbors(kk, jj)

        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        # convolution 1
        net = net + self.c1(mask_ix * net[:,ix])

        # convolution 2
        net = net + self.c2(mask_jx * net[:,jx])

        stop_conv = time.perf_counter()
        with open('times_conv.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_conv - start_conv)]))

        # soft aggregator frames
        start_agg_frames = time.perf_counter()
        net = net + self.agg_kk(net, kk)
        stop_agg_frames = time.perf_counter()
        with open('times_agg_frames.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_agg_frames - start_agg_frames)]))

        # soft aggregator patches
        start_agg_patches = time.perf_counter()
        net = net + self.agg_ij(net, ii*12345 + jj)
        stop_agg_patches = time.perf_counter()
        with open('times_agg_patches.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_agg_patches - start_agg_patches)]))

        # Transition block
        start_transition = time.perf_counter()
        net = self.gru(net)
        stop_transition = time.perf_counter()
        with open('times_transition.txt', 'a+') as file:
            np.savetxt(file, np.array([1000*(stop_transition - start_transition)]))

        # flow and weight factor heads
        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')  # matching network
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')  # context network

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, gradient_bias=False, return_color=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0  # shape [1, 1, 128, 160, 120]
        imap = self.inet(images) / 4.0  # shape [1, 1, 384, 160, 120]

        b, n, c, h, w = fmap.shape

        # bias patch selection towards regions with high gradient
        if gradient_bias:
            g = self.__image_gradient(images)  # shape [1, 1, 159, 119]
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g, coords, 0).view(-1)
            
            ix = torch.argsort(g)
            x = x[:, ix[-patches_per_image:]]
            y = y[:, ix[-patches_per_image:]]

        else:
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, 1).view(b, -1, 128, self.patch_size, self.patch_size)

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, 1).view(b, -1, 3, self.patch_size, self.patch_size)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        #print(f"fmap: {fmap.shape}, gmap: {gmap.shape}, imap: {imap.shape}, patches: {patches.shape}")
        if return_color:
            return fmap, gmap, imap, patches, index, clr

        # fmap: matching features, gmap: patches extracted from matching features
        # imap: patches extracted from context features (1x1), patches: patches indices in the fmap with inverse depth
        # shapes: fmap:[1, 1, 128, img_width/4, img_height/4], gmap: [1, n of patches per frame, 128, patch dim, patch dim],
        # imap: [1, n of patches per frame, 384, 1, 1],  patches: [1, 96, 3 (x,y, and inv depth), patch dim, patch dim]
        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 5  # patch size
        self.patchify = Patchifier(self.P)

        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.
        if disps is not None:
            disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()  # patches with known depth
        Ps = poses  # gt poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))  # patches with unknown depth (initialized at random)

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)

        Gs = SE3.IdentityLike(poses)  # these are the estimated poses

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]  # TODO: pensa a come migliorare questa cosa
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            # coords are the coordinates of the points in patches in frame ii into frame jj
            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                # update our estimate of the poses (GS) and the depths (in patches) using BA
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

