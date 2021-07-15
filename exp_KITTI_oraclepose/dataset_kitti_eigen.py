from __future__ import print_function, division
import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)
sys.path.append('core')

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import copy

import os
import math
import random
from glob import glob
import os.path as osp
import PIL.Image as Image

from core.utils.utils import vls_ins
from core.utils import frame_utils
from torchvision.transforms import ColorJitter
from core.utils.semantic_labels import Label
import time
import cv2
import pickle
from core.utils.frame_utils import readFlowKITTI

def latlonToMercator(lat,lon,scale):
    er = 6378137
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my

def latToScale(lat):
    scale = np.cos(lat * np.pi / 180.0)
    return scale

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

def read_into_numbers(path):
    numbs = list()
    with open(path, 'r') as f:
        numstr = f.readlines()[0].rstrip().split(' ')
        for n in numstr:
            numbs.append(float(n))
    return numbs

def rot_from_axisangle(angs):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    rotx = np.eye(3)
    roty = np.eye(3)
    rotz = np.eye(3)

    rotx[1, 1] = np.cos(angs[0])
    rotx[1, 2] = -np.sin(angs[0])
    rotx[2, 1] = np.sin(angs[0])
    rotx[2, 2] = np.cos(angs[0])

    roty[0, 0] = np.cos(angs[1])
    roty[0, 2] = np.sin(angs[1])
    roty[2, 0] = -np.sin(angs[1])
    roty[2, 2] = np.cos(angs[1])

    rotz[0, 0] = np.cos(angs[2])
    rotz[0, 1] = -np.sin(angs[2])
    rotz[1, 0] = np.sin(angs[2])
    rotz[1, 1] = np.cos(angs[2])

    rot = rotz @ (roty @ rotx)
    return rot

def get_pose(root, seq, index, extrinsic):
    scale = latToScale(read_into_numbers(os.path.join(root, seq, 'oxts/data', "{}.txt".format(str(0).zfill(10))))[0])

    # Pose 1
    oxts_path = os.path.join(root, seq, 'oxts/data', "{}.txt".format(str(index).zfill(10)))
    nums = read_into_numbers(oxts_path)
    mx, my = latlonToMercator(nums[0], nums[1], scale)

    pose1 = np.eye(4)
    t1 = np.array([mx, my, nums[2]])
    ang1 = np.array(nums[3:6])

    pose1[0:3, 3] = t1
    pose1[0:3, 0:3] = rot_from_axisangle(ang1)

    # Pose 2
    oxts_path = os.path.join(root, seq, 'oxts/data', "{}.txt".format(str(index + 1).zfill(10)))
    nums = read_into_numbers(oxts_path)
    mx, my = latlonToMercator(nums[0], nums[1], scale)

    pose2 = np.eye(4)
    t2 = np.array([mx, my, nums[2]])
    ang2 = np.array(nums[3:6])

    pose2[0:3, 3] = t2
    pose2[0:3, 0:3] = rot_from_axisangle(ang2)

    rel_pose = np.linalg.inv(pose2 @ np.linalg.inv(extrinsic)) @ (pose1 @ np.linalg.inv(extrinsic))
    return rel_pose.astype(np.float32)

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

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

def read_deepv2d_pose(deepv2dpose_path):
    # Read Pose from Deepv2d
    posesstr = readlines(deepv2dpose_path)
    poses = list()
    for pstr in posesstr:
        pose = np.zeros([4, 4]).flatten()
        for idx, ele in enumerate(pstr.split(' ')):
            pose[idx] = float(ele)
            if idx == 15:
                break
        pose = np.reshape(pose, [4, 4])
        poses.append(pose)
    pose_deepv2d = poses[3] @ np.linalg.inv(poses[0])
    pose_deepv2d[0:3, 3] = pose_deepv2d[0:3, 3] * 10
    return pose_deepv2d

class KITTI_eigen(data.Dataset):
    def __init__(self, entries, dataset_root, inheight, inwidth, depthgt_root, RANSACPose_root, inDualDirFrames, istrain=True, muteaug=False, isgarg=False):
        super(KITTI_eigen, self).__init__()
        self.istrain = istrain
        self.isgarg = isgarg
        self.muteaug = muteaug
        self.inheight = inheight
        self.inwidth = inwidth
        self.inFrames = inDualDirFrames # Number of frames consumed in one direction during training

        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)

        self.imager_list = list()
        self.poser_list = list()
        self.depthgt_list = list()
        self.intrinsic_list = list()
        self.RANSACPose_list = list()

        self.entries = list()

        for entry in entries:
            seq, index, _ = entry.split(' ')
            index = int(index)

            imgcpath = os.path.join(dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
            depthpath = os.path.join(depthgt_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))

            # Load Intrinsic for each frame
            calib_dir = os.path.join(dataset_root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)

            if not os.path.exists(depthpath):
                continue

            spv_exst = False
            inImgPaths = {0: imgcpath}
            inPoses = {0: np.eye(4)}
            for k in range(1, inDualDirFrames + 1):
                rfrmids = [-k, k]
                for rfrmidx in rfrmids:
                    inImgPath = os.path.join(dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index + rfrmidx).zfill(10)))
                    last_frm = np.sign(rfrmidx) * (np.abs(rfrmidx) - 1)
                    if os.path.exists(inImgPath):
                        spv_exst = True
                        if rfrmidx > 0:
                            incPosePath = os.path.join(RANSACPose_root, seq, 'image_02', "{}.pickle".format(str(index + rfrmidx - 1).zfill(10)))
                            incPose = pickle.load(open(incPosePath, "rb"))
                            inPose = incPose @ inPoses[last_frm]
                        else:
                            incPosePath = os.path.join(RANSACPose_root, seq, 'image_02', "{}.pickle".format(str(index + rfrmidx).zfill(10)))
                            incPose = pickle.load(open(incPosePath, "rb"))
                            inPose = np.linalg.inv(incPose) @ inPoses[last_frm]
                    else:
                        inImgPath = inImgPaths[last_frm]
                        inPose = inPoses[last_frm]
                    inImgPaths[rfrmidx] = copy.deepcopy(inImgPath)
                    inPoses[rfrmidx] = copy.deepcopy(inPose)

            if not spv_exst and istrain:
                # If there does not exist any supervision signal
                continue

            self.intrinsic_list.append(intrinsic)
            self.entries.append(entry)
            self.depthgt_list.append(depthpath)
            self.imager_list.append(inImgPaths)
            self.poser_list.append(inPoses)

        assert len(self.intrinsic_list) == len(self.entries) == len(self.depthgt_list) == len(self.imager_list) == len(self.poser_list)

    def colorjitter(self, img):
        img_auged = np.array(self.photo_aug(img), dtype=np.uint8)
        return img_auged

    def read_imgs(self, index):
        imgr = dict()
        for k in self.imager_list[index].keys():
            imgr[k] = Image.open(self.imager_list[index][k])
        if self.istrain and not self.muteaug:
            inrgb_augmented = self.colorjitter(copy.copy(imgr[0]))
        else:
            inrgb_augmented = copy.copy(imgr[0])

        for k in self.imager_list[index].keys():
            imgr[k] = np.array(imgr[k]).astype(np.float32) / 255.0
        inrgb_augmented = np.array(inrgb_augmented).astype(np.float32) / 255.0
        return imgr, inrgb_augmented

    def __getitem__(self, index):
        imgr, inrgb_augmented = self.read_imgs(index)
        poser = copy.deepcopy(self.poser_list[index])

        depthgt = np.array(Image.open(self.depthgt_list[index])).astype(np.float32) / 256.0
        intrinsic = copy.deepcopy(self.intrinsic_list[index])

        imgr, inrgb_augmented, depthgt, intrinsic = self.aug_crop(imgr, inrgb_augmented, depthgt, intrinsic)
        data_blob = self.wrapup(imgr=imgr, inrgb_augmented=inrgb_augmented, depthgt=depthgt, poser=poser, intrinsic=intrinsic, tag=self.entries[index])
        return data_blob

    def wrapup(self, imgr, inrgb_augmented, depthgt, poser, intrinsic, tag):
        for k in imgr.keys():
            imgr[k] = torch.from_numpy(imgr[k]).permute([2, 0, 1]).float()
        inrgb_augmented = torch.from_numpy(inrgb_augmented).permute([2, 0, 1]).float()
        depthgt = torch.from_numpy(depthgt).unsqueeze(0).float()
        for k in imgr.keys():
            poser[k] = torch.from_numpy(poser[k]).float()
        intrinsic = torch.from_numpy(intrinsic).float()

        data_blob = dict()
        data_blob['imgr'] = imgr
        data_blob['inrgb_augmented'] = inrgb_augmented
        data_blob['depthgt'] = depthgt
        data_blob['poser'] = poser
        data_blob['intrinsic'] = intrinsic
        data_blob['tag'] = tag

        return data_blob

    def crop_img(self, img, left, top, crph, crpw):
        img_cropped = img[top:top+crph, left:left+crpw]
        return img_cropped

    def aug_crop(self, imgr, inrgb_augmented, depthgt, intrinsic):
        if inrgb_augmented.ndim == 3:
            h, w, _ = inrgb_augmented.shape
        else:
            h, w = inrgb_augmented.shape

        crph = self.inheight
        crpw = self.inwidth

        if crph >= h:
            crph = h

        if crpw >= w:
            crpw = w

        if self.istrain:
            left = np.random.randint(0, w - crpw - 1, 1).item()
        else:
            left = int((w - crpw) / 2)
        top = int(h - crph)

        if not self.istrain and self.isgarg:
            crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]).astype(np.int32)
            crop_mask = np.zeros([h, w])
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            depthgt = depthgt * crop_mask
            assert left < crop[2] and left + crpw > crop[3] and top < crop[0]

        intrinsic[0, 2] -= left
        intrinsic[1, 2] -= top

        for k in imgr.keys():
            imgr[k] = self.crop_img(imgr[k], left=left, top=top, crph=crph, crpw=crpw)
        inrgb_augmented = self.crop_img(inrgb_augmented, left=left, top=top, crph=crph, crpw=crpw)

        depthgt = self.crop_img(depthgt, left=left, top=top, crph=crph, crpw=crpw)
        return imgr, inrgb_augmented, depthgt, intrinsic

    def get_gt_flow(self, depth, valid, intrinsic, rel_pose):
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
        return flowgt

    def __len__(self):
        return len(self.entries)

    def debug_pose(self):
        vlsroot = '/media/shengjie/disk1/visualization/imu_accuracy_vls'
        for k in range(500):
            test_idx = np.random.randint(0, len(self.depth_list), 1)[0]

            img1 = Image.open(self.image_list[test_idx][0])
            img2 = Image.open(self.image_list[test_idx][1])

            depth = np.array(Image.open(self.depth_list[test_idx])).astype(np.float32) / 256.0
            depth = torch.from_numpy(depth)

            h, w = depth.shape

            semanticspred = Image.open(self.semantics_list[test_idx])
            semanticspred = semanticspred.resize([w, h], Image.NEAREST)
            semanticspred = np.array(semanticspred)
            semantic_selector = np.ones_like(semanticspred)
            for ll in np.unique(semanticspred).tolist():
                if ll in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
                    semantic_selector[semanticspred == ll] = 0
            semantic_selector = torch.from_numpy(semantic_selector).float()

            intrinsic = torch.from_numpy(self.intrinsic_list[test_idx])
            rel_pose = torch.from_numpy(self.pose_list[test_idx])

            xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
            xx = torch.from_numpy(xx).float()
            yy = torch.from_numpy(yy).float()
            selector = depth > 0

            xxf = xx[selector]
            yyf = yy[selector]
            df = depth[selector]
            pts3d = torch.stack([xxf * df, yyf * df, df, torch.ones_like(df)], dim=0)
            pts3d = torch.inverse(intrinsic) @ pts3d
            pts3d_oview = rel_pose @ pts3d
            pts2d_oview = intrinsic @ pts3d_oview
            pts2d_oview[0, :] = pts2d_oview[0, :] / pts2d_oview[2, :]
            pts2d_oview[1, :] = pts2d_oview[1, :] / pts2d_oview[2, :]

            import matplotlib.pyplot as plt
            cm = plt.get_cmap('magma')
            vmax = 0.15
            tnp = 1 / df.numpy() / vmax
            tnp = cm(tnp)

            fig = plt.figure(figsize=(16, 9))
            fig.add_subplot(2, 1, 1)
            plt.scatter(xxf.numpy(), yyf.numpy(), 1, tnp)
            plt.imshow(img1)

            fig.add_subplot(2, 1, 2)
            plt.scatter(pts2d_oview[0, :].numpy(), pts2d_oview[1, :].numpy(), 1, tnp)
            plt.imshow(img2)

            seq, frmidx, _ = self.entries[test_idx].split(' ')
            plt.savefig(os.path.join(vlsroot, "{}_{}.png".format(seq.split('/')[1], str(frmidx).zfill(10))))
            plt.close()

