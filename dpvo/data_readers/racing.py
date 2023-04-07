import numpy as np
import glob
from torch.utils.data import Dataset
import torch
import cv2
import os.path as osp
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(0)
np.random.seed(0)


#test_split = [
#    "2022-06-08-21-23-03",
#    "2022-03-16-16-12-49",
#    "2022-06-07-20-22-25",
#    "2022-06-10-03-14-16"
#]


#train_split = [
#    "2022-06-08-21-23-03",
#]

#train_split = [
#    "indoor_forward_3_snapdragon_with_gt",
#]

test_split = [
    'indoor_45_3_snapdragon',
    'indoor_45_16_snapdragon',
    'indoor_forward_11_snapdragon',
    'indoor_forward_12_snapdragon',
    'outdoor_forward_9_snapdragon',
    'outdoor_forward_10_snapdragon',
]


def augment_images(images):
    # takes an input a list of images and transform them

    # choose one of the following contrast augmentations
    contrast_aug = iaa.OneOf([
        iaa.GammaContrast((0.67, 1.5)),
        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
        iaa.LinearContrast((0.4, 1.6)),
        iaa.AllChannelsCLAHE(),
    ])

    # combine 2 augmentations from contrast, brightness and sharpen, and apply them in random order
    colors_aug = iaa.SomeOf(2, [
        contrast_aug,
        iaa.MultiplyAndAddToBrightness(mul=(0.67, 1.5), add=(-10, 10)),
        iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.75, 2.0)),
    ], random_order=True)

    # the colors augmentation only apply 50% of the times
    colors_aug = iaa.Sometimes(0.5, colors_aug)

    # combine 2 augmentations from gaussian noise, coarse drop out and salt and pepper, and apply them in random order
    noise_aug = iaa.SomeOf(2, [
        iaa.AdditiveGaussianNoise(scale=(0, 20)),
        iaa.CoarseDropout((0.0, 0.005), size_percent=(0.05, 0.3)),
        iaa.SaltAndPepper((0, 0.02)),
    ], random_order=True)

    # include the cut-out augmentation, where entire regions are being cut out from the image
    noise_aug = iaa.Sequential([
        noise_aug,
        iaa.Cutout(nb_iterations=(0, 3), size=0.15, squared=False, fill_mode="constant", cval=0),
    ])

    # the noise augmentation only apply 50% of the times
    noise_aug = iaa.Sometimes(0.5, noise_aug)

    aug_prov = iaa.Sequential([
        colors_aug,
        noise_aug,
    ])

    # apply the transformation to 50% of the images
    aug = iaa.Sometimes(0.5, aug_prov)

    images_aug = aug(images=images)
    return images_aug


class Racing(Dataset):
    def __init__(self, datapath, n_frames=5, scale=0.5, augmentation=True, validation_size=0.05):
        super(Racing, self).__init__()
        self.datapath = datapath
        self.n_frames = n_frames
        self.info_dataset = self._build_dataset()
        self.dataset_index = self._build_index()
        self.scale = scale
        self.augmentation = augmentation
        self.validation_size = validation_size
        self.validation_index = self._build_validation_index()
        self.validation = False  # change the value when you want to perform validation

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building racing dataset")

        info_dataset = {}
        scenes = glob.glob(osp.join(self.datapath, '*/'))
        scenes = [scene for scene in scenes if osp.basename(osp.dirname(scene)) not in test_split]
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
            if scene in test_split:
                continue
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

    def _build_validation_index(self):
        tot_indices = len(self.dataset_index)
        tot_validation_indices = int(self.validation_size * tot_indices)
        validation_index = (np.random.rand(tot_validation_indices) * tot_indices).astype(int)
        return validation_index

    def __getitem__(self, index):
        if not self.validation:
            # if the index is part of the validation set, randomly resample it
            while index in self.validation_index:
                index = (np.random.rand(1)*len(self.dataset_index)).astype(int)[0]
        scene = self.dataset_index[index][0]
        indices = self.dataset_index[index][1:]
        #print(indices)

        images = []
        for i in indices:
            img = cv2.imread(self.info_dataset[scene]['images'][i])
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w*self.scale), int(h*self.scale)))
            images.append(img)
        if self.augmentation and not self.validation:
            images = augment_images(images)

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


