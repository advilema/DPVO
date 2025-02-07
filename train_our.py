import cv2
import os
import argparse
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import DataLoader
from dpvo.data_readers.racing import Racing

from dpvo.lietorch import SE3

from dpvo.net import VONet
from evaluate_tartan import evaluate as validate
import wandb



def evaluate(net, images, poses, disps, intrinsics):
    # fix poses to gt for first 1k steps
    # so = total_steps < 1000 and args.ckpt is None
    so = False

    poses = SE3(poses).inv()
    traj = net(images, poses, disps, intrinsics, M=1024, STEPS=18, structure_only=so)

    loss = 0.0
    ro_error, tr_error = 0., 0.
    for i, (v, x, y, P1, P2, kl) in enumerate(traj):
        # v: valid, x: estimated coordinates of the patches in frame ii projected into frame jj,
        # y: coordinates_gt: same as x but ground truth. Note that to have actual gt you need a depth map,
        # P1: estimated poses, P2: gt poses, kl: not used.

        e = (x - y).norm(dim=-1)
        e = e.reshape(-1, 9)[(v > 0.5).reshape(-1)].min(dim=-1).values

        N = P1.shape[1]
        ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        ii = ii.reshape(-1).cuda()
        jj = jj.reshape(-1).cuda()

        k = ii != jj
        ii = ii[k]
        jj = jj[k]

        P1 = P1.inv()
        P2 = P2.inv()

        t1 = P1.matrix()[..., :3, 3]
        t2 = P2.matrix()[..., :3, 3]

        s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
        P1 = P1.scale(s.view(1, 1))

        dP = P1[:, ii].inv() * P1[:, jj]
        dG = P2[:, ii].inv() * P2[:, jj]

        e1 = (dP * dG.inv()).log()
        tr = e1[..., 0:3].norm(dim=-1)
        ro = e1[..., 3:6].norm(dim=-1)

        loss += args.flow_weight * e.mean()
        if not so and i >= 2:
            loss += args.pose_weight * (tr.mean() + ro.mean())
            ro_error += ro.mean()
            tr_error += tr.mean()

    return loss, tr_error, ro_error


@torch.no_grad()
def validate(db, net):
    validation_index = db.validation_index
    len_validation = len(validation_index)
    db.validation = True
    losses, tr_errors, ro_errors = [], [], []
    validation_index_indices = (np.random.rand(len_validation//2) * len_validation).astype(int)
    for index in validation_index[validation_index_indices]:
        images, poses, intrinsics = db[index]
        images = images.unsqueeze(0).cuda()
        poses = poses.unsqueeze(0).cuda()
        intrinsics = intrinsics.unsqueeze(0).cuda()
        disps = None
        loss, tr_error, ro_error = evaluate(net, images, poses, disps, intrinsics)
        losses.append(loss.item())
        tr_errors.append(tr_error.item())
        ro_errors.append(ro_error.item())
    db.validation = False
    loss_mean = np.mean(losses)
    tr_errors_mean = np.mean(tr_errors)
    ro_errors_mean = np.mean(ro_errors)
    return loss_mean, tr_errors_mean, ro_errors_mean


def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c

# TODO: add validation
def train(args):
    """ main training loop """

    wandb.init(
        project=args.project_name if args.project_name is not None else args.model_name,
        config={
            'data path': args.datapath,
            'checkpoint': args.ckpt,
            'learning rate': args.lr,
            'total steps': args.steps,
            'n frames': args.n_frames,
            'data augmentation': args.augmentation,
            'flow weight': args.flow_weight,
            'pose_weight': args.pose_weight
        }
    )

    ckpt_every_n_steps = min(int(args.steps / 10), 5000)
    validate_every_n_steps = ckpt_every_n_steps // 5

    db = Racing(args.datapath, n_frames=args.n_frames, scale=1.0, augmentation=args.augmentation, validation_size=args.validation_size)
    train_loader = DataLoader(db, batch_size=args.batch_size, shuffle=True, num_workers=4)

    net = VONet(patch_size=args.patch_size)
    net.train()
    net.cuda()

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    total_steps = 0

    while total_steps < args.steps:
        for data_blob in train_loader:
            images, poses, intrinsics = [x.cuda().float() for x in data_blob]
            #disps = [None for _ in range(len(poses))]
            disps = None
            optimizer.zero_grad()

            loss, tr_error, ro_error = evaluate(net, images, poses, disps, intrinsics)

            loss.backward()

            wandb.log({'loss': loss, 'rotation error': ro_error, 'translation error': tr_error})

            print('**** step: {}'.format(total_steps))
            print('loss: {}'.format(loss))
            print('translation error: {}'.format(tr_error))
            print('rotation error: {}'.format(ro_error))
            print()

            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            if total_steps % ckpt_every_n_steps == 0:
                # validation
                if args.validation_size > 0:
                    v_loss, v_tr_error, v_ro_error = validate(db, net)
                    wandb.log({'validation loss': v_loss, 'validation translation error': v_tr_error, 'validation rotation error': v_ro_error})
                    print('**** VALIDATION')
                    print('loss: {}'.format(v_loss))
                    print('translation error: {}'.format(v_tr_error))
                    print('rotation error: {}'.format(v_ro_error))
                    print()

                torch.cuda.empty_cache()

                # save model checkpoints
                if not os.path.isdir('checkpoints'):
                    os.mkdir('checkpoints')
                PATH = 'checkpoints/%s_%06d.pth' % (args.model_name, total_steps)
                torch.save(net.state_dict(), PATH)

                torch.cuda.empty_cache()
                net.train()
            if total_steps == args.steps:
                break

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', help='name your experiment')
    parser.add_argument('--model_name', default='bla', help='name your experiment')
    parser.add_argument('--datapath', default='/data/scratch/marcomagno/racing', help='path to dataset')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.0)
    parser.add_argument('--validation_size', type=float, default=0.05, help='percentage of data in validation split')

    args = parser.parse_args()

    train(args)
