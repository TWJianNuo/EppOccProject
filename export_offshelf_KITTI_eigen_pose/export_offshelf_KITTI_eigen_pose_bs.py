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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle
from core.utils.frame_utils import readFlowKITTI

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from torchvision.transforms import ColorJitter
from core.utils import frame_utils
from tqdm import tqdm
import copy
from glob import glob

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam):
    pose_imu2cam = np.eye(4)
    pose_imu2cam[0:3, 0:3] = np.reshape(imu2cam['R'], [3, 3])
    pose_imu2cam[0:3, 3] = imu2cam['T']

    pose_velo2cam = np.eye(4)
    pose_velo2cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
    pose_velo2cam[0:3, 3] = velo2cam['T']

    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = cam2cam['R_rect_00'].reshape(3, 3)

    intrinsic = np.eye(4)
    intrinsic[0:3, 0:3] = cam2cam['P_rect_02'].reshape(3, 4)[0:3, 0:3]

    org_intrinsic = np.eye(4)
    org_intrinsic[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
    extrinsic_from_intrinsic = np.linalg.inv(intrinsic) @ org_intrinsic
    extrinsic_from_intrinsic[0:3, 0:3] = np.eye(3)

    extrinsic = extrinsic_from_intrinsic @ R_rect_00 @ pose_velo2cam @ pose_imu2cam

    return intrinsic.astype(np.float32), extrinsic.astype(np.float32)

class KITTI_eigen(data.Dataset):
    def __init__(self, entries, root='datasets/KITTI', ins_root=None, flowPred_root=None, mdPred_root=None):
        super(KITTI_eigen, self).__init__()
        self.root = root
        self.flowPred_root = flowPred_root
        self.mdPred_root = mdPred_root
        self.ins_root = ins_root

        self.image_list = list()
        self.intrinsic_list = list()
        self.inspred_list = list()
        self.flowPred_list = list()
        self.mdPred_list = list()

        self.entries = list()

        for entry in self.remove_dup(entries):
            seq, index = entry.split(' ')
            index = int(index)

            img1path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            img2path = os.path.join(root, seq, 'image_02', 'data', "{}.png".format(str(index + 1).zfill(10)))

            if not os.path.exists(img1path):
                raise Exception("rgb file %s missing" % mdDepth_path)

            mdDepth_path = os.path.join(self.mdPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            if not os.path.exists(mdDepth_path):
                continue

            flowpred_path = os.path.join(self.flowPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
            if not os.path.exists(img1path):
                raise Exception("flow prediction %s missing" % flowpred_path)

            # Load Intrinsic for each frame
            calib_dir = os.path.join(root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            if not os.path.exists(img2path):
                self.image_list.append([img1path, img1path])
            else:
                self.image_list.append([img1path, img2path])

            self.intrinsic_list.append(intrinsic)
            self.entries.append(entry)
            self.flowPred_list.append(flowpred_path)
            self.mdPred_list.append(mdDepth_path)

        assert len(self.intrinsic_list) == len(self.entries) == len(self.image_list)

    def remove_dup(self, entries):
        dupentry = list()
        for entry in entries:
            seq, index, _ = entry.split(' ')
            dupentry.append("{} {}".format(seq, index.zfill(10)))

        removed = list(set(dupentry))
        removed.sort()
        return removed

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        intrinsic = copy.deepcopy(self.intrinsic_list[index])

        mdDepth = np.array(Image.open(self.mdPred_list[index])).astype(np.float32) / 256.0
        flowpred_RAFT, valid_flow = readFlowKITTI(self.flowPred_list[index])
        data_blob = self.wrapup(img1=img1, img2=img2, intrinsic=intrinsic, mdDepth=mdDepth, flowpred_RAFT=flowpred_RAFT, tag=self.entries[index])

        return data_blob

    def wrapup(self, img1, img2, intrinsic, mdDepth, flowpred_RAFT, tag):
        img1 = torch.from_numpy(img1).permute([2, 0, 1]).float()
        img2 = torch.from_numpy(img2).permute([2, 0, 1]).float()
        intrinsic = torch.from_numpy(intrinsic).float()
        mdDepth = torch.from_numpy(mdDepth).unsqueeze(0)
        flowpred_RAFT = torch.from_numpy(flowpred_RAFT).permute([2, 0, 1]).float()

        data_blob = dict()
        data_blob['img1'] = img1
        data_blob['img2'] = img2
        data_blob['intrinsic'] = intrinsic
        data_blob['mdDepth'] = mdDepth
        data_blob['flowpred_RAFT'] = flowpred_RAFT
        data_blob['tag'] = tag

        return data_blob

    def __len__(self):
        return len(self.entries)

def vls_flows(image1, image2, flow_anno, flow_depth, depth, insmap):
    image1np = image1[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
    image2np = image2[0].cpu().permute([1, 2, 0]).numpy().astype(np.uint8)
    depthnp = depth[0].cpu().squeeze().numpy()
    flow_anno_np = flow_anno[0].cpu().numpy()
    flow_depth_np = flow_depth[0].cpu().numpy()
    insmap_np = insmap[0].cpu().squeeze().numpy()

    h, w, _ = image1np.shape
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    selector_anno = (flow_anno_np[0, :, :] != 0) * (depthnp > 0) * (insmap_np == 0)

    flowx = flow_anno_np[0][selector_anno]
    flowy = flow_anno_np[1][selector_anno]

    xxf = xx[selector_anno]
    yyf = yy[selector_anno]
    df = depthnp[selector_anno]

    cm = plt.get_cmap('magma')
    rndcolor = cm(1 / df / 0.15)[:, 0:3]

    selector_depth = (flow_depth_np[0, :, :] != 0) * (depthnp > 0) * (insmap_np == 0)
    flowx_depth = flow_depth_np[0][selector_depth]
    flowy_depth = flow_depth_np[1][selector_depth]

    xxf_depth = xx[selector_depth]
    yyf_depth = yy[selector_depth]
    df_depth = depthnp[selector_depth]
    rndcolor_depth = cm(1 / df_depth / 0.15)[:, 0:3]

    fig = plt.figure(figsize=(16, 9))
    fig.add_subplot(3, 1, 1)
    plt.scatter(xxf, yyf, 3, rndcolor)
    plt.imshow(image1np)

    fig.add_subplot(3, 1, 2)
    plt.scatter(xxf + flowx, yyf + flowy, 3, rndcolor)
    plt.imshow(image2np)

    fig.add_subplot(3, 1, 3)
    plt.scatter(xxf_depth + flowx_depth, yyf_depth + flowy_depth, 3, rndcolor_depth)
    plt.imshow(image2np)
    plt.show()

def depth2flow(depth, valid, intrinsic, rel_pose):
    device = depth.device
    depth = depth.squeeze().cpu().numpy()
    valid = valid.squeeze().cpu().numpy()
    intrinsic = intrinsic.squeeze().cpu().numpy()
    rel_pose = rel_pose.squeeze().cpu().numpy()
    h, w = depth.shape

    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    selector = (valid == 1)

    xxf = xx[selector]
    yyf = yy[selector]
    df = depth[selector]

    pts3d = np.stack([xxf * df, yyf * df, df, np.ones_like(df)], axis=0)
    pts3d = np.linalg.inv(intrinsic) @ pts3d
    pts3d_oview = rel_pose @ pts3d
    pts2d_oview = intrinsic @ pts3d_oview
    pts2d_oview[0, :] = pts2d_oview[0, :] / pts2d_oview[2, :]
    pts2d_oview[1, :] = pts2d_oview[1, :] / pts2d_oview[2, :]
    selector = pts2d_oview[2, :] > 0

    flowgt = np.zeros([h, w, 2])
    flowgt[yyf.astype(np.int)[selector], xxf.astype(np.int)[selector], 0] = pts2d_oview[0, :][selector] - xxf[selector]
    flowgt[yyf.astype(np.int)[selector], xxf.astype(np.int)[selector], 1] = pts2d_oview[1, :][selector] - yyf[selector]
    flowgt = torch.from_numpy(flowgt).permute([2, 0, 1]).unsqueeze(0).cuda(device)
    return flowgt

def depth2scale(pts2d1, pts2d2, intrinsic, R, t, coorespondedDepth):
    intrinsic33 = intrinsic[0:3, 0:3]
    M = intrinsic33 @ R @ np.linalg.inv(intrinsic33)
    delta_t = (intrinsic33 @ t).squeeze()
    minval = 1e-6

    denom = (pts2d2[0, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[0, :], axis=0) @ pts2d1).squeeze()) ** 2 + \
            (pts2d2[1, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[1, :], axis=0) @ pts2d1).squeeze()) ** 2

    selector = (denom > minval)

    rel_d = np.sqrt(
        ((delta_t[0] - pts2d2[0, selector] * delta_t[2]) ** 2 +
         (delta_t[1] - pts2d2[1, selector] * delta_t[2]) ** 2) / denom[selector])
    alpha = np.mean(coorespondedDepth[selector]) / np.mean(rel_d)
    return alpha

def select_scale(scale_md, R, t, pts1_inliers, pts2_inliers, mdDepth_npf, intrinsicnp):
    numres = 49

    divrat = 5
    maxrat = 1
    pos = (np.exp(np.linspace(0, divrat, numres)) - 1) / (np.exp(divrat) - 1) * np.exp(maxrat) + 1
    neg = np.exp(-np.log(pos))
    tot = np.sort(np.concatenate([pos, neg, np.array([1e-5])]))

    scale_md_cand = scale_md * tot

    self_pose = np.eye(4)
    self_pose[0:3, 0:3] = R
    self_pose[0:3, 3:4] = t
    self_pose = np.expand_dims(self_pose, axis=0)
    self_pose = np.repeat(self_pose, axis=0, repeats=tot.shape[0])
    self_pose[:, 0:3, 3] = self_pose[:, 0:3, 3] * np.expand_dims(scale_md_cand, axis=1)

    pts3d = np.stack([pts1_inliers[0, :] * mdDepth_npf, pts1_inliers[1, :] * mdDepth_npf, mdDepth_npf, np.ones_like(mdDepth_npf)])
    pts3d = intrinsicnp @ self_pose @ np.linalg.inv(intrinsicnp) @ np.repeat(np.expand_dims(pts3d, axis=0), axis=0, repeats=tot.shape[0])
    pts3d[:, 0, :] = pts3d[:, 0, :] / pts3d[:, 2, :]
    pts3d[:, 1, :] = pts3d[:, 1, :] / pts3d[:, 2, :]

    loss = np.mean(np.abs(pts3d[:, 0, :] - pts2_inliers[0]), axis=1) + np.mean(np.abs(pts3d[:, 1, :] - pts2_inliers[1]), axis=1)
    np.abs(pts1_inliers[0] - pts2_inliers[0]).mean() + np.abs(pts1_inliers[1] - pts2_inliers[1]).mean()

    best = np.argmin(loss)
    # plt.figure()
    # plt.plot(loss)
    # plt.show()
    return scale_md_cand[best], best

def inf_pose_flow(flow_pr_inf, mdDepth, intrinsic, rndseed, samplenum=50000, ban_exsrh=False):
    intrinsicnp = intrinsic[0].cpu().numpy()
    dummyh = 370
    _, _, h, w = mdDepth.shape

    border_sel = np.zeros([h, w])
    border_sel[int(0.25810811 * dummyh) : int(0.99189189 * dummyh)] = 1

    flow_pr_inf_x = flow_pr_inf[0, 0].cpu().numpy()
    flow_pr_inf_y = flow_pr_inf[0, 1].cpu().numpy()

    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    xx_nf = xx + flow_pr_inf_x
    yy_nf = yy + flow_pr_inf_y

    mdDepth_np = mdDepth.squeeze().cpu().numpy()

    selector = (xx_nf > 0) * (xx_nf < w) * (yy_nf > 0) * (yy_nf < h) * border_sel * (mdDepth_np > 0)
    selector = selector == 1

    if samplenum > np.sum(selector):
        samplenum = np.sum(selector)

    np.random.seed(rndseed)
    rndidx = np.random.randint(0, np.sum(selector), samplenum)

    xx_idx_sel = xx[selector][rndidx]
    yy_idx_sel = yy[selector][rndidx]

    flow_sel_mag = np.mean(np.sqrt(flow_pr_inf_x[yy_idx_sel, xx_idx_sel] ** 2 + flow_pr_inf_y[yy_idx_sel, xx_idx_sel] ** 2))

    pts1 = np.stack([xx_idx_sel, yy_idx_sel], axis=1).astype(np.float)
    pts2 = np.stack([xx_nf[yy_idx_sel, xx_idx_sel], yy_nf[yy_idx_sel, xx_idx_sel]], axis=1).astype(np.float)

    E, inliers = cv2.findEssentialMat(pts1, pts2, focal=intrinsicnp[0,0], pp=(intrinsicnp[0, 2], intrinsicnp[1, 2]), method=cv2.RANSAC, prob=0.99, threshold=0.1)
    cheirality_cnt, R, t, _ = cv2.recoverPose(E, pts1, pts2, focal=intrinsicnp[0, 0], pp=(intrinsicnp[0, 2], intrinsicnp[1, 2]))

    inliers_mask = inliers == 1
    inliers_mask = np.squeeze(inliers_mask, axis=1)
    pts1_inliers = pts1[inliers_mask, :].T
    pts2_inliers = pts2[inliers_mask, :].T

    pts1_inliers = np.concatenate([pts1_inliers, np.ones([1, pts1_inliers.shape[1]])], axis=0)
    pts2_inliers = np.concatenate([pts2_inliers, np.ones([1, pts2_inliers.shape[1]])], axis=0)
    scale_md = depth2scale(pts1_inliers, pts2_inliers, intrinsicnp, R, t, mdDepth_np[selector][rndidx][inliers_mask])

    if R[0, 0] < 0 or R[1, 1] < 0 or R[2, 2] < 0 or t[2] > 0:
        R = np.eye(3)
        t = np.array([[0, 0, -1]]).T
        scale_md = 0
    elif not ban_exsrh:
        scale_md, bestid = select_scale(scale_md, R, t, pts1_inliers, pts2_inliers, mdDepth_np[pts1_inliers[1, :].astype(np.int), pts1_inliers[0, :].astype(np.int)], intrinsicnp)

    # Image.fromarray(flow_to_image(flow_pr_inf[0].cpu().permute([1, 2, 0]).numpy())).show()
    # tensor2disp(torch.from_numpy(selector).unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
    # tensor2disp(mdDepth > 0, vmax=1, viewind=0).show()

    return R, t, scale_md, flow_sel_mag

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

@torch.no_grad()
def validate_RANSAC_odom_relpose(args, eval_loader, samplenum=50000):
    for val_id, data_blob in enumerate(tqdm(eval_loader)):
        intrinsic = data_blob['intrinsic']
        flowpred = data_blob['flowpred_RAFT']
        mdDepth_pred = data_blob['mdDepth']
        tag = data_blob['tag'][0]

        if torch.sum(torch.abs(data_blob['img1'] - data_blob['img2'])) < 1:
            R = np.eye(3)
            t = np.array([[0, 0, -1]]).T
            scale = 0
        else:
            R, t, scale, _ = inf_pose_flow(flowpred, mdDepth_pred, intrinsic, int(val_id + 10), samplenum=samplenum)
        self_pose = np.eye(4)
        self_pose[0:3, 0:3] = R
        self_pose[0:3, 3:4] = t * scale

        seq, frmidx = tag.split(' ')
        exportfold = os.path.join(args.export_root, seq, 'image_02')
        os.makedirs(exportfold, exist_ok=True)

        export_root = os.path.join(exportfold, frmidx.zfill(10) + '.pickle')
        with open(export_root, 'wb') as handle:
            pickle.dump(self_pose, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_odom_entries(args):
    odomentries = list()
    odomseqs = [
        '2011_10_03/2011_10_03_drive_0027_sync',
        '2011_09_30/2011_09_30_drive_0018_sync'
    ]
    for odomseq in odomseqs:
        leftimgs = glob(os.path.join(args.odom_root, odomseq, 'image_02/data', "*.png"))
        for leftimg in leftimgs:
            imgname = os.path.basename(leftimg)
            odomentries.append("{} {} {}".format(odomseq, imgname.rstrip('.png'), 'l'))
    return odomentries

def generate_seqmapping(seqid):
    seqmapping = \
    ['00 2011_10_03_drive_0027 000000 004540',
     '01 2011_10_03_drive_0042 000000 001100',
     "02 2011_10_03_drive_0034 000000 004660",
     "03 2011_09_26_drive_0067 000000 000800",
     "04 2011_09_30_drive_0016 000000 000270",
     "05 2011_09_30_drive_0018 000000 002760",
     "06 2011_09_30_drive_0020 000000 001100",
     "07 2011_09_30_drive_0027 000000 001100",
     "08 2011_09_30_drive_0028 001100 005170",
     "09 2011_09_30_drive_0033 000000 001590",
     "10 2011_09_30_drive_0034 000000 001200"]

    entries = list()
    seqmap = dict()
    for seqm in seqmapping:
        mapentry = dict()
        mapid, seqname, stid, enid = seqm.split(' ')
        if int(mapid) == int(seqid):
            mapentry['mapid'] = int(mapid)
            mapentry['stid'] = int(stid)
            mapentry['enid'] = int(enid)
            seqmap[seqname] = mapentry

            for k in range(int(stid), int(enid)):
                entries.append("{}/{}_sync {} {}".format(seqname[0:10], seqname, str(k).zfill(10), 'l'))
    entries.sort()
    return seqmap, entries

def get_all_entries(args, skipexist=False):
    split_root = os.path.join(exp_root, 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    odom_entries = get_odom_entries(args)

    entries = train_entries + odom_entries
    folds = list()
    for entry in entries:
        seq, idx, _ = entry.split(' ')
        folds.append(seq)
    folds = list(set(folds))

    entries_expand = list()
    for fold in folds:
        pngs = glob(os.path.join(args.dataset_root, fold, 'image_02/data/*.png'))
        for png in pngs:
            frmidx = png.split('/')[-1].split('.')[0]
            entry_expand = "{} {} {}".format(fold, frmidx.zfill(10), 'l')
            entries_expand.append(entry_expand)
    tot_entries = list(set(entries_expand + evaluation_entries))

    if skipexist:
        exportentries = list()
        for e in tot_entries:
            seq, frmidx, _ = e.split(' ')
            exportfold = os.path.join(args.export_root, seq, 'image_02')
            export_root = os.path.join(exportfold, frmidx.zfill(10) + '.pickle')
            try:
                pred_pose = pickle.load(open(export_root, "rb"))
                if pred_pose.shape[0] < 1 or np.ndim(pred_pose) != 3:
                    exportentries.append(e)
            except:
                exportentries.append(e)
    else:
        exportentries = tot_entries

    return exportentries

def export_poses(processid, args, entries):
    interval = np.floor(len(entries) / args.nprocs).astype(np.int).item()
    if processid == args.nprocs - 1:
        stidx = int(interval * processid)
        edidx = len(entries)
    else:
        stidx = int(interval * processid)
        edidx = int(interval * (processid + 1))

    eval_dataset = KITTI_eigen(root=args.dataset_root, entries=entries[stidx : edidx],  flowPred_root=args.flowPred_root, mdPred_root=args.mdPred_root)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False, shuffle=False)
    print("Initial subprocess, from %d to %d, total %d" % (stidx, edidx, len(entries)))
    validate_RANSAC_odom_relpose(args, eval_loader, samplenum=args.samplenum)
    return

def epoposes2txt(args, seqmap, entries, seqid):
    seq = list(seqmap.keys())[0]
    gtposes_sourse = readlines(os.path.join(exp_root, 'kittiodom_gt/poses', "{}.txt".format(str(seqmap[seq[0:21]]['mapid']).zfill(2))))
    gtposes = list()
    for gtpose_src in gtposes_sourse:
        gtpose = np.eye(4).flatten()
        for numstridx, numstr in enumerate(gtpose_src.split(' ')):
            gtpose[numstridx] = float(numstr)
        gtpose = np.reshape(gtpose, [4, 4])
        gtposes.append(gtpose)

    relposes_RANSAC = list()
    for val_id, entry in enumerate(tqdm(entries)):
        seq, frameidx, _ = entry.split(' ')
        try:
            pose_RANSAC_path = os.path.join(args.export_root, entry.split(' ')[0], 'image_02', str(frameidx).zfill(10) + '.pickle')
            pose_RANSAC = pickle.load(open(pose_RANSAC_path, "rb"))
            relposes_RANSAC.append(pose_RANSAC)
        except:
            if val_id == len(entries) - 1:
                continue
            else:
                raise Exception("Missing prediciton")

    if len(relposes_RANSAC) == len(gtposes) - 2:
        relposes_RANSAC.append(np.linalg.inv(gtposes[-1]) @ gtposes[-2])

    accumPose = np.eye(4)
    reconstructed_pose = [accumPose]
    for r in relposes_RANSAC:
        accumPose = r @ accumPose
        reconstructed_pose.append(np.linalg.inv(accumPose))

    exportextpath = os.path.join(args.export_root, str(seqid).zfill(2) + '.txt')
    with open(exportextpath, "w") as text_file:
        for k, r in enumerate(reconstructed_pose):
            r = r[0:3, :]
            printstr = ""
            for n in r.flatten():
                printstr += "{:.6e} ".format(n)
            printstr = printstr[0:-1]
            if k < len(reconstructed_pose) - 1:
                text_file.write(printstr + '\n')
            else:
                text_file.write(printstr)

def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y) / np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


class KittiEvalOdom():
    # ----------------------------------------------------------------------
    # poses: N,4,4
    # pose: 4,4
    # ----------------------------------------------------------------------
    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def loadPoses(self, file_name):
        # ----------------------------------------------------------------------
        # Each line in the file should follow one of the following structures
        # (1) idx pose(3x4 matrix in terms of 12 numbers)
        # (2) pose(3x4 matrix in terms of 12 numbers)
        # ----------------------------------------------------------------------
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ")]
            withIdx = int(len(line_split) == 13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row * 4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectory_distances(self, poses):
        # ----------------------------------------------------------------------
        # poses: dictionary: [frame_idx: pose]
        # ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    def rotation_error(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def last_frame_from_segment_length(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10

        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(dist, first_frame, len_)

                # ----------------------------------------------------------------------
                # Continue if sequence not long enough
                # ----------------------------------------------------------------------
                if last_frame == -1 or not (last_frame in poses_result.keys()) or not (
                        first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
                # compute rotational and translational errors
                # ----------------------------------------------------------------------
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # ----------------------------------------------------------------------
                # compute speed
                # ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    def save_sequence_errors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write + "\n")
        fp.close()

    def compute_overall_err(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def plotPath(self, seq, poses_gt, poses_result):
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num = -1

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_" + (seq)
        plt.savefig(self.plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0)
        # plt.show()

    def scale_optimization(self, gt, pred):
        """ Optimize scaling factor
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            new_pred (4x4 array dict): predicted poses after optimization
        """
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in pred:
            pose_pred = pred[i]
            pose_ref = gt[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    def eval(self, gt_txt, result_txt, seq, naln=False):
        # gt_dir: the directory of groundtruth poses txt
        # results_dir: the directory of predicted poses txt
        self.plot_path_dir = os.path.dirname(result_txt)

        self.gt_txt = gt_txt

        poses_result = self.loadPoses(result_txt)
        poses_gt = self.loadPoses(self.gt_txt)

        # Pose alignment to first frame
        idx_0 = sorted(list(poses_result.keys()))[0]
        pred_0 = poses_result[idx_0]
        gt_0 = poses_gt[idx_0]
        for cnt in poses_result:
            poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
            poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

        if not naln:
            # get XYZ
            xyz_gt = []
            xyz_result = []
            for cnt in poses_result:
                xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
                xyz_result.append([poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
            xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
            xyz_result = np.asarray(xyz_result).transpose(1, 0)

            r, t, scale = umeyama_alignment(xyz_result, xyz_gt, True)

            align_transformation = np.eye(4)
            align_transformation[:3:, :3] = r
            align_transformation[:3, 3] = t


            for cnt in poses_result:
                poses_result[cnt][:3, 3] *= scale
                poses_result[cnt] = align_transformation @ poses_result[cnt]

        # ----------------------------------------------------------------------
        # compute sequence errors
        # ----------------------------------------------------------------------
        seq_err = self.calc_sequence_errors(poses_gt, poses_result)

        # ----------------------------------------------------------------------
        # compute overall error
        # ----------------------------------------------------------------------
        ave_t_err, ave_r_err = self.compute_overall_err(seq_err)

        if not naln:
            print("Aligned, Sequence:%s, Translational error : %f , Rotational error (deg/100m): %f" % (str(seq).zfill(2), ave_t_err * 100, ave_r_err / np.pi * 180 * 100))
            self.plotPath(str(seq).zfill(2), poses_gt, poses_result)
        else:
            print("Unaligned, Sequence:%s, Translational error : %f , Rotational error (deg/100m): %f" % (str(seq).zfill(2), ave_t_err * 100, ave_r_err / np.pi * 180 * 100))
            self.plotPath(str(seq).zfill(2) + '_noalign', poses_gt, poses_result)
        return ave_t_err, ave_r_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--odom_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--export_root', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--samplenum', type=int, default=50000)
    parser.add_argument('--nprocs', type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    entries = get_all_entries(args)
    mp.spawn(export_poses, nprocs=args.nprocs, args=(args, entries))

    seqids = [0, 5]
    for seqid in seqids:
        seqmap, val_entries = generate_seqmapping(seqid)
        epoposes2txt(args, seqmap, val_entries, seqid)

    eval_tool = KittiEvalOdom()
    for seqid in seqids:
        result_txt = os.path.join(args.export_root, "{}.txt".format(str(seqid).zfill(2)))

        seqmap, val_entries = generate_seqmapping(seqid)
        gt_txt = os.path.join(exp_root, 'kittiodom_gt/poses', "{}.txt".format(str(seqid).zfill(2)))
        eval_tool.eval(gt_txt, result_txt, seq=seqid, naln=False)

    for seqid in seqids:
        result_txt = os.path.join(args.export_root, "{}.txt".format(str(seqid).zfill(2)))

        seqmap, val_entries = generate_seqmapping(seqid)
        gt_txt = os.path.join(exp_root, 'kittiodom_gt/poses', "{}.txt".format(str(seqid).zfill(2)))
        eval_tool.eval(gt_txt, result_txt, seq=seqid, naln=True)