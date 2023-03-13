import numpy as np
import glob
from torch.utils.data import Dataset
import torch
import cv2
import os.path as osp



#test_split = [
#    "2022-06-08-21-23-03",
#    "2022-03-16-16-12-49",
#    "2022-06-07-20-22-25",
#    "2022-06-10-03-14-16"
#]


train_split = [
    "indoor_forward_5_snapdragon_with_gt",
]


class Racing(Dataset):
    def __init__(self, datapath, n_frames=5, scale=0.5):
        super(Racing, self).__init__()
        self.datapath = datapath
        self.n_frames = n_frames
        self.info_dataset = self._build_dataset()
        self.dataset_index = self._build_index()
        self.scale = scale

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building racing dataset")

        info_dataset = {}
        scenes = glob.glob(osp.join(self.datapath, '*/'))
        scenes = [scene for scene in scenes if osp.basename(osp.dirname(scene)) in train_split]
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'images/*.png')))
            poses = np.loadtxt(osp.join(scene, 'poses.csv'), delimiter=',')
            intrinsics = np.loadtxt(osp.join(scene, 'intrinsics.csv'), delimiter=',')
            intrinsics = np.repeat(np.reshape(intrinsics, (1, intrinsics.shape[0])), poses.shape[0], axis=0)

            scene = '/'.join(scene.split('/'))
            info_dataset[scene] = {'images': images, 'poses': poses, 'intrinsics': intrinsics}

        return info_dataset

    def _build_index(self):
        dataset_index = []
        for scene in self.info_dataset:
            n_frames_scene = self.info_dataset[scene]['poses'].shape[0]
            for i in range(n_frames_scene):
                indices = [i]
                idx = 1
                while i + idx < n_frames_scene and idx < self.n_frames:
                    indices.append(i + idx)
                    idx += 1
                if len(indices) == self.n_frames:
                    dataset_index.append([scene, *indices])
        return dataset_index

    def __getitem__(self, index):
        scene = self.dataset_index[index][0]
        indices = self.dataset_index[index][1:]
        #print(indices)

        images = []
        for i in indices:
            img = cv2.imread(self.info_dataset[scene]['images'][i])
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*self.scale), int(h*self.scale)))
            images.append(img)

        poses = [self.info_dataset[scene]['poses'][i] for i in indices]
        intrinsics = [self.info_dataset[scene]['intrinsics'][i] for i in indices]

        images = np.stack(images).astype(np.float32)
        #print(images.shape)
        poses = np.stack(poses).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)

        return images, poses, intrinsics

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self


