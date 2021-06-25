from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import random
import numpy as np
from numpy.random import default_rng
import cv2
import PIL.Image as Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from numba import njit, prange
import math
import tqdm

from core.utils.frame_utils import read_gen, readFlowVRKitti
from exp_VRKitti.dataset_VRKitti2 import VirtualKITTI2
from core.utils.utils import tensor2disp, tensor2rgb

def read_splits(project_rootdir):
    split_root = os.path.join(project_rootdir, 'exp_VRKitti', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'training_split.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'evaluation_split.txt'), 'r')]
    return train_entries, evaluation_entries

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

# @njit
def occ_detect_c_numba_debug(epp1, epp2, pts2dsrch_v1_batch, pts2d_v1_batch, pts2d_v2_batch, seach_distance, out_range_selecor, occ_selector, min_dist, w, h, sx, sy):
    for x in range(w):
        for y in range(h):
            if sx == x and sy == y:
                if seach_distance[y, x] > min_dist:
                    pts_ss = bresenham(x0=x, y0=y, x1=float(pts2dsrch_v1_batch[y, x, 0]), y1=float(pts2dsrch_v1_batch[y, x, 1]), w=w, h=h)

                    pts_occ_rec_v1 = list()
                    pts_occ_rec_v2 = list()
                    for i in range(1, pts_ss.shape[0]):
                        pts_dst_v1 = pts_ss[i]
                        pts_src_v1 = pts2d_v1_batch[y, x, :]
                        vec1_v1 = pts_src_v1 - epp1
                        vec1_v1 = vec1_v1 / np.sqrt(np.sum(vec1_v1 ** 2))
                        vec2_v1 = pts_dst_v1 - pts_src_v1
                        vec2_v1 = vec2_v1 / np.sqrt(np.sum(vec2_v1 ** 2))
                        dot_prodt1 = np.sum(vec1_v1 * vec2_v1)

                        pts_dst_v2 = pts2d_v2_batch[int(pts_ss[i, 1].item()), int(pts_ss[i, 0].item())]
                        pts_src_v2 = pts2d_v2_batch[y, x, :]
                        vec1_v2 = pts_src_v2 - epp2
                        vec1_v2 = vec1_v2 / np.sqrt(np.sum(vec1_v2 ** 2))
                        vec2_v2 = pts_dst_v2 - pts_src_v2
                        vec2_v2 = vec2_v2 / np.sqrt(np.sum(vec2_v2 ** 2))
                        dot_prodt2 = np.sum(vec1_v2 * vec2_v2)

                        if dot_prodt1 * dot_prodt2 < 0:
                            pts_occ_rec_v1.append(pts_dst_v1)
                            pts_occ_rec_v2.append(pts_dst_v2)

                return pts_ss, pts_occ_rec_v1, pts_occ_rec_v2

def occ_detect_c_debug(intrinsic, pose, depthmap, sx, sy):
    h, w = depthmap.shape
    min_dist = 1
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
    return occ_detect_c_numba_debug(epp1.astype(np.float32), epp2.astype(np.float32), pts2dsrch_v1_batch.astype(np.float32), pts2d_v1_batch.astype(np.float32), pts2d_v2_batch.astype(np.float32), seach_distance.astype(np.float32), out_range_selecor, occ_selector, min_dist, w=w, h=h, sx=sx, sy=sy)


@njit
def occ_detect_c_numba(epp1, epp2, pts2dsrch_v1_batch, pts2d_v1_batch, pts2d_v2_batch, seach_distance, out_range_selecor, occ_selector, minsr_dist, minoc_dist, w, h):
    for x in range(w):
        for y in range(h):
            if seach_distance[y, x] > minsr_dist:
                pts_ss = bresenham(x0=x, y0=y, x1=float(pts2dsrch_v1_batch[y, x, 0]), y1=float(pts2dsrch_v1_batch[y, x, 1]), w=w, h=h)

                for i in range(1, pts_ss.shape[0]):
                    if not out_range_selecor[int(pts_ss[i, 1].item()), int(pts_ss[i, 0].item())]:
                        pts_dst_v1 = pts_ss[i]
                        pts_src_v1 = pts2d_v1_batch[y, x, :]
                        vec1_v1 = pts_src_v1 - epp1
                        vec1_v1 = vec1_v1 / np.sqrt(np.sum(vec1_v1 ** 2))
                        vec2_v1 = pts_dst_v1 - pts_src_v1
                        vec2_v1 = vec2_v1 / np.sqrt(np.sum(vec2_v1 ** 2))
                        dot_prodt1 = np.sum(vec1_v1 * vec2_v1)

                        pts_dst_v2 = pts2d_v2_batch[int(pts_ss[i, 1].item()), int(pts_ss[i, 0].item())]
                        pts_src_v2 = pts2d_v2_batch[y, x, :]
                        vec1_v2 = pts_src_v2 - epp2
                        vec1_v2 = vec1_v2 / np.sqrt(np.sum(vec1_v2 ** 2))
                        vec2_v2 = pts_dst_v2 - pts_src_v2
                        vec2_v2 = vec2_v2 / np.sqrt(np.sum(vec2_v2 ** 2))
                        dot_prodt2 = np.sum(vec1_v2 * vec2_v2)

                        if dot_prodt1 * dot_prodt2 < 0:
                            if np.sqrt(np.sum((pts_dst_v2 - pts_src_v2) ** 2)) < minoc_dist:
                                occ_selector[int(pts_ss[i, 1].item()), int(pts_ss[i, 0].item())] = True

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


if __name__ == '__main__':
    vrkittiroot = '/media/shengjie/disk1/data/virtual_kitti_organized'
    project_rootdir = '/home/shengjie/Documents/supporting_projects/RAFT'
    vlsroot = '/media/shengjie/disk1/visualization/epp_occ_vrkitti_backward'

    os.makedirs(vlsroot, exist_ok=True)

    bz = 1
    width = 1242
    height = 375
    maxinsnum = 200
    insnum = 5

    train_entries, evaluation_entries = read_splits(project_rootdir)

    vrkitti2 = VirtualKITTI2(root=vrkittiroot, entries=train_entries, inheight=375, inwidth=1242, istrain=False)
    vrkitti2loader = DataLoader(vrkitti2, batch_size=bz, pin_memory=False, shuffle=False, num_workers=0, drop_last=True)

    for case_idx in tqdm.tqdm(range(0, len(train_entries), 10)):

        data_blob = vrkitti2loader.dataset.__getitem__(case_idx)

        img1 = data_blob['img2'].unsqueeze(0)
        img2 = data_blob['img1'].unsqueeze(0)
        flowmap = data_blob['flowmap'].unsqueeze(0)
        intrinsic = data_blob['intrinsic'].unsqueeze(0)
        poses = data_blob['poses'].unsqueeze(0)
        tag = data_blob['tag']
        senename, seqname, frmidx = tag.split('_')

        uppercase = list(senename.upper()[0])
        senename = list(senename)
        senename[0] = uppercase[0]
        senename = "".join(senename)
        flowmap_back = os.path.join('/media/shengjie/disk1/data/virtual_kitti', 'vkitti_2.0.3_backwardFlow/{}/{}/frames/backwardFlow/Camera_0/backwardFlow_{}.png'.format(senename, seqname, str(int(frmidx) + 1).zfill(5)))

        if not os.path.exists(flowmap_back):
            continue
        try:
            flowmap_back = np.array(readFlowVRKitti(flowmap_back)).astype(np.float32)
            flowmap_back = torch.from_numpy(flowmap_back).permute([2, 0, 1]).unsqueeze(0)
        except:
            continue

        pose_np = np.linalg.inv(poses.squeeze()[0].cpu().numpy())
        intrinsic_np = intrinsic.squeeze().cpu().numpy()
        depthmap_np = os.path.join(vrkittiroot, "Scene{}".format(str(int(senename[5:7])).zfill(2)), seqname, 'frames', 'depth', 'Camera_0', "depth_{}.png".format(str(int(frmidx) + 1).zfill(5)))
        depthmap_np = Image.open(depthmap_np)
        depthmap_np = np.array(depthmap_np).astype(np.float32) / 100.0

        tmp = torch.clone(flowmap_back)
        flowmap_back = flowmap
        flowmap = tmp


        xx, yy = np.meshgrid(list(range(width)), list(range(height)))
        xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0)
        yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0)
        pts2d = torch.cat([xx, yy], dim=1)
        pts2d_f = pts2d + flowmap

        pts2d_f_normed = torch.clone(pts2d_f)
        pts2d_f_normed[:, 0] = 2*pts2d_f_normed[:, 0]/(width-1) - 1
        pts2d_f_normed[:, 1] = 2*pts2d_f_normed[:, 1]/(height-1) - 1
        pts2d_f_normed = pts2d_f_normed.permute([0, 2, 3, 1])
        flowmap_back_b = F.grid_sample(flowmap_back, pts2d_f_normed, align_corners=True)
        flowmap_diff = flowmap + flowmap_back_b
        flowmap_diff = torch.sum(flowmap_diff.abs(), dim=1, keepdim=True)

        # Inference
        occ_selector1 = occ_detect_c(intrinsic=intrinsic_np, pose=pose_np, depthmap=depthmap_np, minoc_dist=3)
        # occ_selector2 = occ_detect_c(intrinsic=intrinsic_np, pose=pose_np, depthmap=depthmap_np, minoc_dist=3e10)

        fig1 = tensor2rgb(data_blob['img1'].unsqueeze(0), viewind=0)
        fig2 = tensor2rgb(data_blob['img2'].unsqueeze(0), viewind=0)
        fig3 = tensor2disp(torch.from_numpy(occ_selector1).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0)
        # fig4 = tensor2disp(torch.from_numpy(occ_selector2).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0)
        fig5 = tensor2disp(flowmap_diff > 1, vmax=1, viewind=0)
        # fig_combined = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3), np.array(fig4), np.array(fig5)], axis=0)
        fig_combined = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3), np.array(fig5)], axis=0)
        Image.fromarray(fig_combined).save(os.path.join(vlsroot, "{}.png".format(tag)))

        # cam_org_3d = np.array([[0, 0, 0, 1]]).T
        # epp1 = intrinsic_np @ np.linalg.inv(pose_np) @ cam_org_3d # Epipole on frame 1, projection of camera 2
        # epp1[0, 0] = epp1[0, 0] / epp1[2, 0]
        # epp1[1, 0] = epp1[1, 0] / epp1[2, 0]
        # epp2 = intrinsic_np @ pose_np @ cam_org_3d # Epipole on frame 2, projection of camera 1
        # epp2[0, 0] = epp2[0, 0] / epp2[2, 0]
        # epp2[1, 0] = epp2[1, 0] / epp2[2, 0]
        #
        # # Check the correctness of the Epp Geometry
        # sm_num = 100
        # # rndx = np.random.randint(0, width, 1).item()
        # # rndy = np.random.randint(0, height, 1).item()
        # rndx = 858
        # rndy = 168
        # rndd = depthmap_np[rndy, rndx]
        # log_range = np.linspace(-10, 10, sm_num - 1)
        # log_range = np.concatenate([log_range, np.zeros([1])], axis=0)
        # log_range = np.sort(log_range)
        # rndd_expanded = np.exp(np.log(rndd) + log_range)
        # rnd_pts3d_v1 = np.stack([np.ones([sm_num]) * rndx * rndd_expanded, np.ones([sm_num]) * rndy * rndd_expanded, rndd_expanded, np.ones([sm_num])], axis=1).T
        # rnd_pts3d_v2 = intrinsic_np @ pose_np @ np.linalg.inv(intrinsic_np) @ rnd_pts3d_v1
        # rnd_pts2d_v2 = np.copy(rnd_pts3d_v2)
        # rnd_pts2d_v2[0, :] = rnd_pts2d_v2[0, :] / rnd_pts2d_v2[2, :]
        # rnd_pts2d_v2[1, :] = rnd_pts2d_v2[1, :] / rnd_pts2d_v2[2, :]
        # selector_pos = rnd_pts2d_v2[2, :] > 0
        # rnd_pts2d_v2 = rnd_pts2d_v2[:, selector_pos]
        #
        # # Check the correctness of the Scale Ambiguity
        # scale_np = np.sqrt(np.sum(pose_np[0:3, 3] ** 2))
        # scale_np_expand = np.exp(np.log(scale_np) + log_range)
        # pose_np_expand = np.repeat(np.expand_dims(pose_np, axis=0), sm_num, axis=0)
        # pose_np_expand[:, 0:3, 3] = pose_np_expand[:, 0:3, 3] / np.sqrt(np.sum(pose_np_expand[:, 0:3, 3] ** 2, axis=1, keepdims=True)) * np.expand_dims(scale_np_expand, axis=1)
        # rnd_pts3d_v1_s = np.stack([[rndx * rndd], [rndy * rndd], [rndd], [1]], axis=1).T
        # rnd_pts3d_v2_s = np.expand_dims(intrinsic_np, axis=0) @ pose_np_expand @ np.expand_dims(np.linalg.inv(intrinsic_np) @ rnd_pts3d_v1_s, axis=0)
        # rnd_pts3d_v2_s = np.squeeze(rnd_pts3d_v2_s).T
        # rnd_pts2d_v2_s = np.copy(rnd_pts3d_v2_s)
        # rnd_pts2d_v2_s[0, :] = rnd_pts2d_v2_s[0, :] / rnd_pts2d_v2_s[2, :]
        # rnd_pts2d_v2_s[1, :] = rnd_pts2d_v2_s[1, :] / rnd_pts2d_v2_s[2, :]
        # selector_pos_s = rnd_pts2d_v2_s[2, :] > 0
        # rnd_pts2d_v2_s = rnd_pts2d_v2_s[:, selector_pos_s]
        #
        # # Compute the Pure Rotation Movement at frame T+1
        # pose_np_pr = np.copy(pose_np)
        # pose_np_pr[0:3, 3] = 0
        # rnd_pts3d_v1_pr = np.stack([[rndx * rndd], [rndy * rndd], [rndd], [1]], axis=1).T
        # rnd_pts3d_v2_pr = intrinsic_np @ pose_np_pr @ np.linalg.inv(intrinsic_np) @ rnd_pts3d_v1_pr
        # rnd_pts2d_v2_pr = np.copy(rnd_pts3d_v2_pr)
        # rnd_pts2d_v2_pr[0, 0] = rnd_pts2d_v2_pr[0, 0] / rnd_pts2d_v2_pr[2, 0]
        # rnd_pts2d_v2_pr[1, 0] = rnd_pts2d_v2_pr[1, 0] / rnd_pts2d_v2_pr[2, 0]
        #
        # # Compute the Pure Rotation Movement at frame T
        # rnd_pts3d_v1_pr_bck = intrinsic_np @ np.linalg.inv(pose_np_pr) @ np.linalg.inv(intrinsic_np) @ intrinsic_np @ pose_np @ np.linalg.inv(intrinsic_np) @ rnd_pts3d_v1_pr
        # rnd_pts2d_v1_pr_bck = np.copy(rnd_pts3d_v1_pr_bck)
        # rnd_pts2d_v1_pr_bck[0, 0] = rnd_pts2d_v1_pr_bck[0, 0] / rnd_pts2d_v1_pr_bck[2, 0]
        # rnd_pts2d_v1_pr_bck[1, 0] = rnd_pts2d_v1_pr_bck[1, 0] / rnd_pts2d_v1_pr_bck[2, 0]
        #
        # # Draw Epipole on two images
        # zero_idx = np.argmin(np.abs(log_range[selector_pos]))
        # cm = plt.get_cmap('seismic')
        # vls_color = cm((log_range + 10) / 20)
        # vls_color = np.copy(vls_color[selector_pos, :])
        # vls_color_s = cm((log_range + 10) / 20)
        # vls_color_s = np.copy(vls_color_s[selector_pos_s, :])
        #
        # # Debug
        # pts_search, pts_occ_rec_v1, pts_occ_rec_v2 = occ_detect_c_debug(intrinsic=intrinsic_np, pose=pose_np, depthmap=depthmap_np, sx=rndx, sy=rndy)
        # pts_occ_rec_v1 = np.stack(pts_occ_rec_v1, axis=0)
        # pts_occ_rec_v2 = np.stack(pts_occ_rec_v2, axis=0)
        #
        # fig, axs = plt.subplots(4, 1, figsize=(16, 9))
        # axs[0].scatter(pts_occ_rec_v1[:, 0], pts_occ_rec_v1[:, 1], s=3, c='g')
        # axs[0].scatter([epp1[0, 0]], [epp1[1, 0]], s=10, c='r')
        # axs[0].scatter([rndx], [rndy], s=10, c='c')
        # axs[0].imshow(tensor2rgb(data_blob['img1'].unsqueeze(0), viewind=0))
        # axs[1].scatter([epp2[0, 0]], [epp2[1, 0]], s=10, c='r')
        # axs[1].scatter([rnd_pts2d_v2[0, zero_idx]], [rnd_pts2d_v2[1, zero_idx]], s=10, c='c')
        # axs[1].scatter(pts_occ_rec_v2[:, 0], pts_occ_rec_v2[:, 1], s=3, c='g')
        # axs[1].imshow(tensor2rgb(data_blob['img2'].unsqueeze(0), viewind=0))
        # axs[2].imshow(tensor2disp(torch.from_numpy(occ_selector1).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0))
        # axs[3].imshow(tensor2disp(torch.from_numpy(occ_selector2).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0))
        #
        # # Draw Rasterized Line on two images
        # pts_search = bresenham(x0=rndx, y0=rndy, x1=rnd_pts2d_v1_pr_bck[0, 0], y1=rnd_pts2d_v1_pr_bck[1, 0])
        #
        # fig, axs = plt.subplots(3, 1, figsize=(16, 9))
        # axs[0].scatter(pts_search[:, 0], pts_search[:, 1], s=3, c='g')
        # # axs[0].scatter(pts_occ_rec_v1[:, 0], pts_occ_rec_v1[:, 1], s=3, c='g')
        # axs[0].scatter([epp1[0, 0]], [epp1[1, 0]], s=10, c='r')
        # axs[0].scatter([rndx], [rndy], s=10, c='g')
        # axs[0].scatter([rnd_pts2d_v1_pr_bck[0, 0]], [rnd_pts2d_v1_pr_bck[1, 0]], s=10, c='c')
        # axs[0].imshow(tensor2rgb(data_blob['img1'].unsqueeze(0), viewind=0))
        # axs[1].scatter(rnd_pts2d_v2[0, :], rnd_pts2d_v2[1, :], s=6, c=vls_color)
        # # axs[1].scatter(pts_occ_rec_v2[:, 0], pts_occ_rec_v2[:, 1], s=3, c='g')
        # axs[1].scatter([epp2[0, 0]], [epp2[1, 0]], s=10, c='r')
        # axs[1].scatter([rnd_pts2d_v2[0, zero_idx]], [rnd_pts2d_v2[1, zero_idx]], s=10, c='g')
        # axs[1].scatter([rnd_pts2d_v2_pr[0, 0]], [rnd_pts2d_v2_pr[1, 0]], s=10, c='c')
        # axs[1].imshow(tensor2rgb(data_blob['img2'].unsqueeze(0), viewind=0))
        # axs[2].scatter(rnd_pts2d_v2_s[0, :], rnd_pts2d_v2_s[1, :], s=6, c=vls_color)
        # axs[2].scatter([epp2[0, 0]], [epp2[1, 0]], s=10, c='r')
        # axs[2].scatter([rnd_pts2d_v2[0, zero_idx]], [rnd_pts2d_v2[1, zero_idx]], s=10, c='g')
        # axs[2].imshow(tensor2rgb(data_blob['img2'].unsqueeze(0), viewind=0))