from __future__ import print_function, division
import os, sys
prj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, prj_root)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from dev_selfocc_cuda_KITTI.dataset_KITTI_eigen import KITTI_eigen

import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from tqdm import tqdm

from numba import njit, prange
import math

@njit
def bresenham(x0, y0, x1, y1, w, h):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    count = 0
    pts = np.zeros((math.ceil(dx) + 1, 2), dtype=np.float32)
    for x in range(math.ceil(dx) + 1):
        comx = x0 + x * xx + y * yx
        comy = y0 + x * xy + y * yy
        if comx < 0 or comx >= w or comy < 0 or comy >= h:
            break
        pts[x, 0] = comx
        pts[x, 1] = comy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

        count += 1
    assert count >= 1
    return pts[0:count, :]

@njit
def occ_detect_c_numba(epp1, epp2, pts2dsrch_v1_batch, pts2d_v1_batch, pts2d_v2_batch, seach_distance, out_range_selecor, occ_selector, minsr_dist, minoc_dist, w, h):
    for x in range(w):
        for y in range(h):
            if seach_distance[y, x] > minsr_dist:
                # pts_ss = bresenham(x0=x, y0=y, x1=float(pts2dsrch_v1_batch[y, x, 0]), y1=float(pts2dsrch_v1_batch[y, x, 1]), w=w, h=h)

                dx = float(pts2dsrch_v1_batch[y, x, 0]) - x
                dy = float(pts2dsrch_v1_batch[y, x, 1]) - y

                xsign = 1 if dx > 0 else -1
                ysign = 1 if dy > 0 else -1

                dx = abs(dx)
                dy = abs(dy)

                if dx > dy:
                    xx, xy, yx, yy = xsign, 0, 0, ysign
                else:
                    dx, dy = dy, dx
                    xx, xy, yx, yy = 0, ysign, xsign, 0

                D = 2 * dy - dx
                cty = 0

                for ctx in range(math.ceil(dx) + 1):
                    comx = x + ctx * xx + cty * yx
                    comy = y + ctx * xy + cty * yy
                    if comx < 0 or comx >= w or comy < 0 or comy >= h:
                        break
                    pts_dst_v1 = np.array([comx, comy])
                    if D >= 0:
                        cty += 1
                        D -= 2 * dx
                    D += 2 * dy

                    if not occ_selector[int(comy), int(comx)] and ctx > 0:
                        pts_src_v1 = pts2d_v1_batch[y, x, :]
                        vec1_v1 = pts_src_v1 - epp1
                        vec1_v1 = vec1_v1 / np.sqrt(np.sum(vec1_v1 ** 2))
                        vec2_v1 = pts_dst_v1 - pts_src_v1
                        vec2_v1 = vec2_v1 / np.sqrt(np.sum(vec2_v1 ** 2))
                        dot_prodt1 = np.sum(vec1_v1 * vec2_v1)

                        pts_dst_v2 = pts2d_v2_batch[int(comy), int(comx)]
                        pts_src_v2 = pts2d_v2_batch[y, x, :]
                        vec1_v2 = pts_src_v2 - epp2
                        vec1_v2 = vec1_v2 / np.sqrt(np.sum(vec1_v2 ** 2))
                        vec2_v2 = pts_dst_v2 - pts_src_v2
                        vec2_v2 = vec2_v2 / np.sqrt(np.sum(vec2_v2 ** 2))
                        dot_prodt2 = np.sum(vec1_v2 * vec2_v2)

                        if dot_prodt1 * dot_prodt2 < 0:
                            if np.sqrt(np.sum((pts_dst_v2 - pts_src_v2) ** 2)) < minoc_dist:
                                occ_selector[int(comy), int(comx)] = True

    return occ_selector

def occ_detect_c(intrinsic, pose, depthmap, minoc_dist=3):
    h, w = depthmap.shape
    minsr_dist = 1
    pose_pr = np.copy(pose)
    pose_pr[0:3, 3] = 0

    cam_org_3d = np.array([[0, 0, 0, 1]]).T
    epp1 = intrinsic @ np.linalg.inv(pose) @ cam_org_3d # Epipole on frame 1, projection of camera 2
    epp1[0, 0] = epp1[0, 0] / epp1[2, 0]
    epp1[1, 0] = epp1[1, 0] / epp1[2, 0]
    epp1 = epp1[0:2, 0]
    epp2 = intrinsic @ pose @ cam_org_3d # Epipole on frame 2, projection of camera 1
    epp2[0, 0] = epp2[0, 0] / epp2[2, 0]
    epp2[1, 0] = epp2[1, 0] / epp2[2, 0]
    epp2 = epp2[0:2, 0]

    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    ones = np.ones_like(xx)
    pts3d_v1_batch = np.stack([xx * depthmap, yy * depthmap, depthmap, ones], axis=2)
    pts3d_v1_batch = np.expand_dims(pts3d_v1_batch, axis=3)
    pts2d_v1_batch = np.stack([xx, yy], axis=2)

    pts3d_v2_batch = intrinsic @ pose @ np.linalg.inv(intrinsic) @ pts3d_v1_batch
    pts2d_v2_batch = np.copy(pts3d_v2_batch)
    pts2d_v2_batch[:, :, 0, 0] = pts2d_v2_batch[:, :, 0, 0] / pts2d_v2_batch[:, :, 2, 0]
    pts2d_v2_batch[:, :, 1, 0] = pts2d_v2_batch[:, :, 1, 0] / pts2d_v2_batch[:, :, 2, 0]
    pts2d_v2_batch = pts2d_v2_batch[:, :, 0:2, 0]
    out_range_selecor = (pts3d_v2_batch[:, :, 2, 0] > 0) * (pts2d_v2_batch[:, :, 0] >= 0) * (pts2d_v2_batch[:, :, 0] <= w - 1) * (pts2d_v2_batch[:, :, 1] >= 0) * (pts2d_v2_batch[:, :, 1] <= h - 1)
    out_range_selecor = (out_range_selecor == 0)
    occ_selector = np.copy(out_range_selecor)

    pts2dsrch_v1_batch = intrinsic @ np.linalg.inv(pose_pr) @ np.linalg.inv(intrinsic) @ intrinsic @ pose @ np.linalg.inv(intrinsic) @ pts3d_v1_batch
    pts2dsrch_v1_batch[:, :, 0, 0] = pts2dsrch_v1_batch[:, :, 0, 0] / pts2dsrch_v1_batch[:, :, 2, 0]
    pts2dsrch_v1_batch[:, :, 1, 0] = pts2dsrch_v1_batch[:, :, 1, 0] / pts2dsrch_v1_batch[:, :, 2, 0]
    pts2dsrch_v1_batch = pts2dsrch_v1_batch[:, :, 0:2, 0]

    seach_distance = np.sqrt(np.sum((pts2dsrch_v1_batch - pts2d_v1_batch) ** 2, axis=2))
    return occ_detect_c_numba(epp1.astype(np.float32), epp2.astype(np.float32), pts2dsrch_v1_batch.astype(np.float32), pts2d_v1_batch.astype(np.float32), pts2d_v2_batch.astype(np.float32), seach_distance.astype(np.float32), out_range_selecor, occ_selector, minsr_dist, minoc_dist, w=w, h=h)

def read_splits():
    split_root = os.path.join(exp_root, 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    return train_entries, evaluation_entries

def train(gpu, args):
    args.gpu = gpu
    train_entries, evaluation_entries = read_splits()

    kitti_dataset = KITTI_eigen(root=args.dataset_root, inheight=args.inheight, inwidth=args.inwidth, entries=evaluation_entries, maxinsnum=args.maxinsnum,
                                depth_root=args.depth_root, depthvls_root=args.depthvlsgt_root, ins_root=args.ins_root, mdPred_root=args.mdPred_root,
                                RANSACPose_root=args.RANSACPose_root)
    kitti_dataloader = data.DataLoader(kitti_dataset, batch_size=1, pin_memory=True, num_workers=0, drop_last=False, shuffle=False)

    for i_batch, data_blob in tqdm(enumerate(kitti_dataloader)):
        image1 = data_blob['img1'] / 255.0
        image2 = data_blob['img2'] / 255.0
        intrinsic = data_blob['intrinsic']
        posepred = data_blob['posepred']
        mD_pred = data_blob['mdDepth_pred']

        seq, frmidx = data_blob['tag'][0].split(' ')

        pose_np = posepred.squeeze()[0].cpu().numpy()
        intrinsic_np = intrinsic.squeeze().cpu().numpy()
        depthmap_np = mD_pred.squeeze().cpu().numpy()

        # Inference
        occ_selector = occ_detect_c(intrinsic=intrinsic_np, pose=pose_np, depthmap=depthmap_np, minoc_dist=5)

        fig1 = tensor2rgb(image1, viewind=0)
        fig2 = tensor2rgb(image2, viewind=0)
        fig3 = tensor2disp(torch.from_numpy(occ_selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0)
        fig4 = tensor2disp(1 / mD_pred, vmax=0.15, viewind=0)
        fig_combined = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig4), np.array(fig3)], axis=0)
        Image.fromarray(fig_combined).save(os.path.join(args.vls_root, "{}_{}.png".format(seq.split('/')[1], frmidx)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inheight', type=int, default=320)
    parser.add_argument('--inwidth', type=int, default=960)
    parser.add_argument('--maxinsnum', type=int, default=50)
    parser.add_argument('--baninsmap', action='store_true')

    parser.add_argument('--vls_root', type=str)
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--semantics_root', type=str)
    parser.add_argument('--depth_root', type=str)
    parser.add_argument('--depthvlsgt_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--RANSACPose_root', type=str)
    parser.add_argument('--ins_root', type=str)

    args = parser.parse_args()

    os.makedirs(args.vls_root, exist_ok=True)

    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.cuda.empty_cache()

    train(0, args)