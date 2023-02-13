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
from dpvo.stream import image_stream, video_stream, image_stream1
import pandas as pd
import random
import numpy as np



SKIP = 0


def make_result_folder(filename, args, cfg):
    result_folder = "racing/results/" + filename
    if args.run_name:
        result_folder += '_' + args.run_name
    result_folder += "_ppf" + str(cfg.PATCHES_PER_FRAME)
    result_folder += "_pl" + str(cfg.PATCH_LIFETIME)
    result_folder += "_sk" + str(args.skip)
    result_folder += "_st" + str(args.stride)
    result_folder += "_se" + str(args.seed)
    return result_folder


def save_results(args, cfg):
    poses_df = pd.DataFrame(poses, columns=['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    filename = os.path.basename(args.imagedir)
    filename_ts = filename
    if '_undistorted' in filename:
        filename_ts = filename_ts[:-12]
    ts_path = os.path.join(args.ts, filename_ts + '.csv')
    ts_df = pd.read_csv(ts_path)
    ts_iter = range(args.skip, len(ts_df['ts']), args.stride)
    timestamps = ts_df['ts'][ts_iter]
    timestamps = timestamps.to_numpy()
    poses_df.insert(0, 'timestamps', timestamps)
    result_folder = make_result_folder(filename, args, cfg)
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    traj_estimated_path = os.path.join(result_folder, 'stamped_traj_estimate.txt')
    poses_df.to_csv(traj_estimated_path, index=False, sep=' ', header=False)
    gt_df = pd.read_csv(os.path.join(args.ground_truth_poses, filename_ts + '.csv'))
    gt_path = os.path.join(result_folder, 'stamped_groundtruth.txt')
    gt_df.to_csv(gt_path, index=False, sep=' ', header=False)



def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False, undistort=False):

    slam = None
    queue = Queue(maxsize=8)
    patches_per_frame = []

    if os.path.isdir(imagedir):
        if undistort:
            reader = Process(target=image_stream1, args=(queue, imagedir, calib, stride, skip))
        else:
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
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)
        if t > 0:
            patches_per_frame.append(slam.ii.shape[0])

        image = image.cuda()
        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    reader.join()
    print()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--ts', type=str, default='racing/ts') # path to ground_truth timestamps
    parser.add_argument('--ground_truth_poses', type=str, default='racing/ground_truth_poses')
    parser.add_argument('--run_name', type=str)  # this name will be added to the filename to create the poses file name
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--undistort', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    poses, tstamps = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit,
                         args.undistort)

    # save poses in a csv file
    save_results(args, cfg)
