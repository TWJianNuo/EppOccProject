from __future__ import print_function, division
import os, sys
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from exp_KITTI_oraclepose.dataset_kitti_eigen import KITTI_eigen
from exp_KITTI_oraclepose.eppnet.EppNet import EppNet

from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.autograd import Variable
from tqdm import tqdm

def read_splits(project_rootdir):
    split_root = os.path.join(project_rootdir, 'export_offshelf_KITTI_eigen_pose', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    return train_entries

def retur_drawing(depthmap):
    cm = plt.get_cmap('magma')
    h, w = depthmap.shape
    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    selector = depthmap > 0
    xxf = xx[selector]
    yyf = yy[selector]
    df = depthmap[selector]
    colors = cm(1 / df / 0.15)
    return xxf, yyf, colors

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))

    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def compute_meansurments(gt, pred):
    selector = (gt > 0) * (pred > 0)
    gtf = gt[selector]
    predf = pred[selector]
    performance = compute_errors(gtf, predf)
    return performance

def print_metric(metric, tag):
    print(tag + ':')
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(metric[i]), end='')
    print('{:7.3f}'.format(metric[8]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', type=str)
    parser.add_argument('--my_raw_depthmap', type=str)
    parser.add_argument('--official_raw_depthmap', type=str)
    parser.add_argument('--semidense_depthmap1', type=str)
    parser.add_argument('--semidense_depthmap2', type=str)
    parser.add_argument('--my_clean_depthmap', type=str)
    parser.add_argument('--vls_root', type=str)

    args = parser.parse_args()
    os.makedirs(args.vls_root, exist_ok=True)

    torch.manual_seed(1234)
    np.random.seed(1234)

    train_entries = read_splits(project_rootdir)
    np.random.shuffle(train_entries)

    maxvlsnum = 0
    max_test_num = 20

    count = 0

    perform_my_raw1 = list()
    perform_official_raw1 = list()
    perform_my_clean1 = list()

    perform_my_raw2 = list()
    perform_official_raw2 = list()
    perform_my_clean2 = list()

    for _, entry in enumerate(train_entries):
        seq, frmidx, _ = entry.split(' ')

        my_raw_path = os.path.join(args.my_raw_depthmap, seq, 'image_02', '{}.png'.format(frmidx.zfill(10)))
        my_clean_path = os.path.join(args.my_clean_depthmap, seq, 'image_02', '{}.png'.format(frmidx.zfill(10)))
        official_raw_path = os.path.join(args.official_raw_depthmap, 'train', seq.split('/')[1], 'proj_depth/velodyne_raw/image_02', '{}.png'.format(frmidx.zfill(10)))
        semidense_gt_path1 = os.path.join(args.semidense_depthmap1, 'train', seq.split('/')[1], 'proj_depth/groundtruth/image_02', '{}.png'.format(frmidx.zfill(10)))
        semidense_gt_path2 = os.path.join(args.semidense_depthmap2, seq, 'image_02', '{}.png'.format(frmidx.zfill(10)))

        if os.path.exists(my_raw_path) and os.path.exists(my_clean_path) and os.path.exists(official_raw_path) and os.path.exists(semidense_gt_path1):
            my_raw = Image.open(my_raw_path)
            my_raw = np.array(my_raw).astype(np.float32) / 256.0

            my_clean = Image.open(my_clean_path)
            my_clean = np.array(my_clean).astype(np.float32) / 256.0

            official_raw = Image.open(official_raw_path)
            official_raw = np.array(official_raw).astype(np.float32) / 256.0

            semidense_gt1 = Image.open(semidense_gt_path1)
            semidense_gt1 = np.array(semidense_gt1).astype(np.float32) / 256.0

            semidense_gt2 = Image.open(semidense_gt_path2)
            semidense_gt2 = np.array(semidense_gt2).astype(np.float32) / 256.0

            assert my_raw.shape == my_clean.shape == official_raw.shape == semidense_gt1.shape == semidense_gt2.shape

            if count < maxvlsnum:
                fig, axs = plt.subplots(3, 1, figsize=(16, 9))
                xxf, yyf, colors = retur_drawing(my_clean)
                axs[0].scatter(xxf, yyf, 0.7, colors)
                xxf, yyf, colors = retur_drawing(my_raw)
                axs[1].scatter(xxf, yyf, 0.7, colors)
                xxf, yyf, colors = retur_drawing(official_raw)
                axs[2].scatter(xxf, yyf, 0.7, colors)
                plt.savefig(os.path.join(args.vls_root, '{}_{}.png'.format(seq.split('/')[1], frmidx.zfill(10))))
                plt.close()

            if count > max_test_num:
                break

            perform_my_clean1.append(compute_meansurments(gt=semidense_gt1, pred=my_clean))
            perform_my_raw1.append(compute_meansurments(gt=semidense_gt1, pred=my_raw))
            perform_official_raw1.append(compute_meansurments(gt=semidense_gt1, pred=official_raw))

            perform_my_clean2.append(compute_meansurments(gt=semidense_gt2, pred=my_clean))
            perform_my_raw2.append(compute_meansurments(gt=semidense_gt2, pred=my_raw))
            perform_official_raw2.append(compute_meansurments(gt=semidense_gt2, pred=official_raw))
            count = count + 1

    perform_my_raw1 = np.stack(perform_my_raw1, axis=0)
    perform_my_raw1 = np.mean(perform_my_raw1, axis=0)
    perform_official_raw1 = np.stack(perform_official_raw1, axis=0)
    perform_official_raw1 = np.mean(perform_official_raw1, axis=0)
    perform_my_clean1 = np.stack(perform_my_clean1, axis=0)
    perform_my_clean1 = np.mean(perform_my_clean1, axis=0)

    print_metric(perform_my_clean1, 'perform_my_clean1')
    print_metric(perform_my_raw1, 'perform_my_raw1')
    print_metric(perform_official_raw1, 'perform_official_raw1')

    print("=================================")

    perform_my_raw2 = np.stack(perform_my_raw2, axis=0)
    perform_my_raw2 = np.mean(perform_my_raw2, axis=0)
    perform_official_raw2 = np.stack(perform_official_raw2, axis=0)
    perform_official_raw2 = np.mean(perform_official_raw2, axis=0)
    perform_my_clean2 = np.stack(perform_my_clean2, axis=0)
    perform_my_clean2 = np.mean(perform_my_clean2, axis=0)

    print_metric(perform_my_clean2, 'perform_my_clean2')
    print_metric(perform_my_raw2, 'perform_my_raw2')
    print_metric(perform_official_raw2, 'perform_official_raw2')
