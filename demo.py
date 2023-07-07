import cv2
import numpy as np
import glob
import os.path as osp
import os
import torch
from multiprocessing import Process, Queue

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.stream import image_stream, video_stream
import pandas as pd
import random
import numpy as np
import time
from dpvo.lietorch import SE3
from dpvo import lietorch

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)



SKIP = 0


def make_result_folder(filename):
    result_folder = os.path.join('results', '')
    if args.run_name:
        result_folder += args.run_name + '_'
    network_name = os.path.basename(args.network)
    network_name = network_name.replace('.pth', '')
    result_folder += os.path.basename(network_name)
    result_folder += "_ppf" + str(cfg.PATCHES_PER_FRAME)
    result_folder += "_rw" + str(cfg.REMOVAL_WINDOW)
    result_folder += "_ow" + str(cfg.OPTIMIZATION_WINDOW)
    result_folder += "_pl" + str(cfg.PATCH_LIFETIME)
    if cfg.GRADIENT_BIAS:
        result_folder += "_gb"
    result_folder += "_sk" + str(args.skip)
    result_folder += "_st" + str(args.stride)
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    if cfg.MOTION_MODEL == 'GT':
        filename += '_gt' + str(cfg.NOISE)
    elif cfg.MOTION_MODEL != 'GT' and cfg.MOTION_MODEL != 'DAMPED_LINEAR':
        filename += '_nomotion'

    result_folder = os.path.join(result_folder, filename)

    return result_folder


def save_results():
    poses_df = pd.DataFrame(poses, columns=['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    ts_vec = np.genfromtxt(args.ts)
    ts_iter = range(args.skip, len(ts_vec), args.stride)
    timestamps = ts_vec[ts_iter]
    poses_df.insert(0, 'timestamps', timestamps)
    result_folder = make_result_folder(filename)
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    traj_estimated_path = os.path.join(result_folder, 'stamped_traj_estimate.txt')
    poses_df.to_csv(traj_estimated_path, index=False, sep=' ', header=False)

    # save ground truth
    if args.ground_truth_poses is not None:
        gt = np.genfromtxt(args.ground_truth_poses, delimiter=',')
        gt = gt[ts_iter]
        gt_path = os.path.join(result_folder, 'stamped_groundtruth.txt')
        timestamps = np.expand_dims(timestamps, axis=1)
        gt = np.append(timestamps, gt, axis=1)
        np.savetxt(gt_path, gt, delimiter=' ')



def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):
    times = []

    slam = None
    queue = Queue(maxsize=8)
    patches_per_frame = []

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        print(f'Frame: {t}')
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            ts_vec = np.genfromtxt(args.ts)
            ts_iter = range(args.skip, len(ts_vec), args.stride)
            gt = np.genfromtxt(args.ground_truth_poses, delimiter=',')
            gt = gt[ts_iter]
            position_noise = np.random.rand(len(gt), 3)*cfg.NOISE - cfg.NOISE/2
            gt[:,:3] += position_noise
            slam = DPVO(cfg, network, poses_gt=gt, ht=image.shape[1], wd=image.shape[2], viz=viz, enable_timing=timeit)
        if t > 0:
            patches_per_frame.append(slam.ii.shape[0])
        image = image.cuda()
        intrinsics = intrinsics.cuda()

        start = time.perf_counter()
        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)
        stop = time.perf_counter()
        times.append(1000*(stop - start))

    for _ in range(12):
        slam.update()

    reader.join()
    print()
    np.savetxt('times_tot.txt', times)

    poses, tstamps = slam.terminate()
    return poses, tstamps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--datafolder', type=str, default='/data/scratch/marcomagno/datasets/uzh_fpv/indoor_forward_3_snapdragon_with_gt/images')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--ts', type=str, ) # path to ground_truth timestamps
    parser.add_argument('--ground_truth_poses', type=str)
    parser.add_argument('--run_name', type=str)  # this name will be added to the filename
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--cache', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    if args.datafolder is not None:
        filename = os.path.basename(args.datafolder)
        args.imagedir = os.path.join(args.datafolder, 'images')
        args.calib = os.path.join(args.datafolder, 'intrinsics.csv')
        args.ts = os.path.join(args.datafolder, 'ts.csv')
        args.ground_truth_poses = os.path.join(args.datafolder, 'poses.csv')
        if not os.path.isfile(args.ground_truth_poses):
            # if there is no ground truth file we assign None
            args.ground_truth_poses = None
    else:
        filename = os.path.basename(args.imagedir)

    if args.cache and os.path.isdir(make_result_folder()):
        print('{} already processed before, skipping it'.format(filename))
        exit()
    else:
        print(filename)

    poses, _ = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)

    # save poses in a csv file
    save_results()
