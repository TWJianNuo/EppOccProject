from __future__ import print_function, division
import os, sys
prj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, prj_root)
sys.path.append('core')

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from core.utils.flow_viz import flow_to_image
from core.utils.utils import InputPadder, forward_interpolate, tensor2disp, tensor2rgb, vls_ins
from tqdm import tqdm
import copy
from glob import glob
import random
from core import self_occ_detector
from scipy.stats.stats import pearsonr
import torch.multiprocessing as mp
import pickle

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

def readlines(filename):
    with open(filename, 'r') as f:
        filenames = f.readlines()
    return filenames

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

def get_all_entries(args):
    split_root = os.path.join(exp_root, 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'train_files.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'test_files.txt'), 'r')]
    odom_entries = get_odom_entries(args)
    return train_entries

def get_relpose(args, seq, index, ri):
    relpose = np.eye(4)
    if ri < 0:
        ks = list(range(index + ri, index))
        ks.sort()
        ks = ks[::-1]
    else:
        ks = list(range(index, index + ri))
        ks.sort()

    for k in ks:
        cpose_path = os.path.join(args.pose_root, seq, 'image_02', "{}.pickle".format(str(k).zfill(10)))
        cpose = pickle.load(open(cpose_path, "rb"))
        if ri < 0:
            relpose = np.linalg.inv(cpose) @ relpose
        else:
            # relpose = relpose @ cpose
            relpose = cpose @ relpose

    return relpose

def calc_err_curve_tmpre(processid, args, entries, dovls=False, debug_mode=False):
    interval = np.floor(len(entries) / args.nprocs).astype(np.int).item()
    if processid == args.nprocs - 1:
        stidx = int(interval * processid)
        edidx = len(entries)
    else:
        stidx = int(interval * processid)
        edidx = int(interval * (processid + 1))
    entries = entries[stidx:edidx]
    if len(entries) == 0:
        return

    num_trial = 100
    idx_range = args.idx_range
    rindices = list(range(-idx_range, idx_range + 1))

    log_range = np.linspace(-args.maxdiff, args.maxdiff, num_trial - 1)
    log_range = list(log_range)
    log_range.append(0)
    log_range.sort()
    log_range = np.array(log_range)
    rat_range = np.exp(log_range)
    keyidx = np.argmin(np.abs(rat_range - 1))

    sod = self_occ_detector.apply

    for _, entry in enumerate(tqdm(entries)):
        seq, index, _ = entry.split(' ')
        index = int(index)

        imgrec = dict()
        imgreconrec = dict()
        imgreconmDrec = dict()
        imgreconPermrec = dict()
        pts2drec = dict()
        occmasks = dict()

        corr_list_rec = dict()
        depth_list_rec = dict()
        for kk in range(1, idx_range + 1):
            corr_list_rec[kk] = list()
            depth_list_rec[kk] = list()

        skipf = False
        for ri in rindices:
            imgopath = os.path.join(args.dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index + ri).zfill(10)))
            cpose_path = os.path.join(args.pose_root, seq, 'image_02', "{}.pickle".format(str(index + ri).zfill(10)))
            if not os.path.exists(imgopath) or not os.path.exists(cpose_path):
                skipf = True
        if skipf:
            continue

        imgcpath = os.path.join(args.dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index).zfill(10)))
        imgc = Image.open(imgcpath)
        try:
            imgc.verify()
        except:
            continue
        imgc = Image.open(imgcpath)

        mdGt_path = os.path.join(args.mdGt_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
        mdGt = Image.open(mdGt_path)
        mdGt = np.array(mdGt).astype(np.float32) / 256.0

        mdPred_path = os.path.join(args.mdPred_root, seq, 'image_02', "{}.png".format(str(index).zfill(10)))
        mdPred = Image.open(mdPred_path)
        mdPred = np.array(mdPred).astype(np.float32) / 256.0

        h, w = mdGt.shape
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        ones = np.ones_like(xx)
        pts3d_v1_batch = np.stack([xx * mdGt, yy * mdGt, mdGt, ones], axis=2)
        pts3d_v1_batch = np.expand_dims(pts3d_v1_batch, axis=3)

        pts3dmD_v1_batch = np.stack([xx * mdPred, yy * mdPred, mdPred, ones], axis=2)
        pts3dmD_v1_batch = np.expand_dims(pts3dmD_v1_batch, axis=3)

        perm_selector = (mdGt > 0)
        if np.sum(perm_selector) == 0:
            continue

        imgo_integflag = True
        for ri in rindices:
            if ri == 0:
                imgrec[ri] = imgc
                imgreconmDrec[ri] = imgc
                imgc_data = np.copy(np.array(imgc))
                imgreconPermrec[ri] = np.repeat(np.expand_dims(imgc_data[perm_selector, :].astype(np.float32), axis=0), axis=0, repeats=num_trial)
                pts2drec[ri] = np.stack([xx, yy], axis=2)
                occmasks[ri] = np.ones([h, w], dtype=np.bool)
                continue

            imgopath = os.path.join(args.dataset_root, seq, 'image_02', 'data', "{}.png".format(str(index + ri).zfill(10)))
            try:
                imgo = Image.open(imgopath)
                imgo.verify()
            except:
                imgo_integflag = False
                break
            imgo = Image.open(imgopath)
            imgrec[ri] = imgo

            # Load Intrinsic for each frame
            calib_dir = os.path.join(args.dataset_root, seq.split('/')[0])

            cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
            velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
            imu2cam = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            intrinsic, extrinsic = get_intrinsic_extrinsic(cam2cam, velo2cam, imu2cam)
            relpose = get_relpose(args, seq, index, ri)

            occmask = sod(torch.from_numpy(intrinsic).unsqueeze(0).cuda().float(),
                               torch.from_numpy(relpose).unsqueeze(0).cuda().float(),
                               torch.from_numpy(mdPred).unsqueeze(0).unsqueeze(0).cuda(), float(1e10))
            occmask_np = occmask.squeeze().cpu().numpy()
            occmask_np = occmask_np == 0
            occmasks[ri] = occmask_np

            pts3d_v2_batch = intrinsic @ relpose @ np.linalg.inv(intrinsic) @ pts3d_v1_batch
            pts2d_v2_batch = np.copy(pts3d_v2_batch)
            pts2d_v2_batch[:, :, 0, 0] = pts2d_v2_batch[:, :, 0, 0] / (pts2d_v2_batch[:, :, 2, 0] + 1e-10)
            pts2d_v2_batch[:, :, 1, 0] = pts2d_v2_batch[:, :, 1, 0] / (pts2d_v2_batch[:, :, 2, 0] + 1e-10)
            pts2d_v2_batch = pts2d_v2_batch[:, :, 0:2, 0]
            pts2drec[ri] = pts2d_v2_batch

            pts2d_v2_batch_t = torch.from_numpy(pts2d_v2_batch).float()
            pts2d_v2_batch_t[:, :, 0] = (pts2d_v2_batch_t[:, :, 0] / (w - 1) - 0.5) * 2
            pts2d_v2_batch_t[:, :, 1] = (pts2d_v2_batch_t[:, :, 1] / (h - 1) - 0.5) * 2
            pts2d_v2_batch_t = pts2d_v2_batch_t.unsqueeze(0)
            imgo_data = np.copy(np.array(imgo))
            imgo_t = torch.from_numpy(imgo_data).float().permute([2, 0, 1]).unsqueeze(0)
            imgo_t_recon = F.grid_sample(imgo_t, pts2d_v2_batch_t, align_corners=False)
            imgo_t_recon = imgo_t_recon * (occmask == 0).float().cpu() * (torch.from_numpy(mdGt).unsqueeze(0).unsqueeze(0) > 0).float()
            imgo_t_reconf = Image.fromarray(imgo_t_recon.squeeze().permute([1, 2, 0]).numpy().astype(np.uint8))
            imgreconrec[ri] = imgo_t_reconf

            pts3dmD_v2_batch = intrinsic @ relpose @ np.linalg.inv(intrinsic) @ pts3dmD_v1_batch
            pts2dmD_v2_batch = np.copy(pts3dmD_v2_batch)
            pts2dmD_v2_batch[:, :, 0, 0] = pts2dmD_v2_batch[:, :, 0, 0] / (pts2dmD_v2_batch[:, :, 2, 0] + 1e-10)
            pts2dmD_v2_batch[:, :, 1, 0] = pts2dmD_v2_batch[:, :, 1, 0] / (pts2dmD_v2_batch[:, :, 2, 0] + 1e-10)
            pts2dmD_v2_batch = pts2dmD_v2_batch[:, :, 0:2, 0]

            pts2dmD_v2_batch_t = torch.from_numpy(pts2dmD_v2_batch).float()
            pts2dmD_v2_batch_t[:, :, 0] = (pts2dmD_v2_batch_t[:, :, 0] / (w - 1) - 0.5) * 2
            pts2dmD_v2_batch_t[:, :, 1] = (pts2dmD_v2_batch_t[:, :, 1] / (h - 1) - 0.5) * 2
            pts2dmD_v2_batch_t = pts2dmD_v2_batch_t.unsqueeze(0)
            imgo_t_recon = F.grid_sample(imgo_t, pts2dmD_v2_batch_t, align_corners=False)
            imgo_t_recon = imgo_t_recon * (occmask == 0).float().cpu()
            imgo_t_reconf = Image.fromarray(imgo_t_recon.squeeze().permute([1, 2, 0]).numpy().astype(np.uint8))
            imgreconmDrec[ri] = imgo_t_reconf

            perm_occ_selector = (occmask_np == 0)[perm_selector] # 1 mark occluded, will be assigned negative value
            xxf = np.repeat(np.expand_dims(xx[perm_selector], axis=0), axis=0, repeats=num_trial)
            yyf = np.repeat(np.expand_dims(yy[perm_selector], axis=0), axis=0, repeats=num_trial)
            df = np.repeat(np.expand_dims(mdGt[perm_selector], axis=0), axis=0, repeats=num_trial) * np.expand_dims(rat_range, axis=1)
            pts3dperm_v1_batch = np.stack([xxf * df, yyf * df, df, np.ones_like(df)], axis=2)
            pts3dperm_v1_batch = np.expand_dims(pts3dperm_v1_batch, axis=3)

            pts3dperm_v2_batch = np.expand_dims(intrinsic @ relpose @ np.linalg.inv(intrinsic), axis=[0, 1]) @ pts3dperm_v1_batch
            pts2dperm_v2_batch = np.copy(pts3dperm_v2_batch)
            pts2dperm_v2_batch[:, :, 0, 0] = pts2dperm_v2_batch[:, :, 0, 0] / (pts2dperm_v2_batch[:, :, 2, 0] + 1e-10)
            pts2dperm_v2_batch[:, :, 1, 0] = pts2dperm_v2_batch[:, :, 1, 0] / (pts2dperm_v2_batch[:, :, 2, 0] + 1e-10)
            pts2dperm_v2_batch = pts2dperm_v2_batch[:, :, 0:2, 0]

            pts3dmD_v2_batch_t = torch.from_numpy(pts2dperm_v2_batch).float()
            pts3dmD_v2_batch_t[:, :, 0] = (pts3dmD_v2_batch_t[:, :, 0] / (w - 1) - 0.5) * 2
            pts3dmD_v2_batch_t[:, :, 1] = (pts3dmD_v2_batch_t[:, :, 1] / (h - 1) - 0.5) * 2
            pts3dmD_v2_batch_t = pts3dmD_v2_batch_t.unsqueeze(0)
            imgo_t_recon_perm = F.grid_sample(imgo_t, pts3dmD_v2_batch_t, align_corners=False)
            imgo_t_recon_perm = imgo_t_recon_perm.squeeze().permute([1, 2, 0]).float().numpy()
            imgo_t_recon_perm[:, perm_occ_selector, :] = -1
            imgreconPermrec[ri] = imgo_t_recon_perm

        if not imgo_integflag:
            continue

        # Compute the correlation for each one of them
        accum_abserr_rec = dict()
        accum_abserr_count_rec = dict()
        accum_abserr = np.zeros([num_trial, np.sum(perm_selector)])
        accum_abserr_count = np.zeros([num_trial, np.sum(perm_selector)])
        for kk in range(1, idx_range + 1):
            frm_to_add = [kk, -kk]
            for frm in frm_to_add:
                abserr = np.mean(np.abs(imgreconPermrec[frm] - imgreconPermrec[0]), axis=2)
                abserr_count = (np.sum(imgreconPermrec[frm], axis=2) > 0).astype(np.float32)

                accum_abserr += abserr * abserr_count
                accum_abserr_count += abserr_count
            accum_abserr_rec[kk] = accum_abserr / (accum_abserr_count + 1e-5)
            accum_abserr_count_rec[kk] = accum_abserr_count

        # Store the results in a gigantic list
        df = mdGt[perm_selector]
        for kk in range(1, idx_range + 1):
            for kkk in range(np.sum(perm_selector)):
                if np.sum(accum_abserr_count_rec[kk][:, kkk] == 0) == 0:
                    if np.sum(np.isnan(accum_abserr_rec[kk][:, kkk])) > 0 or np.sum(np.isinf(accum_abserr_rec[kk][:, kkk])) > 0:
                        continue
                    c1, _ = pearsonr(accum_abserr_rec[kk][keyidx::, kkk], rat_range[keyidx::])
                    c2, _ = pearsonr(accum_abserr_rec[kk][0:keyidx + 1, kkk], rat_range[0:keyidx + 1])
                    c = (c1 + -c2) / 2
                    if np.isnan(c):
                        continue
                    corr_list_rec[kk].append(c)
                    depth_list_rec[kk].append(df[kkk])

        svfold_corr = os.path.join(args.tmp_restorage, seq, "corr")
        svfold_depth = os.path.join(args.tmp_restorage, seq, "depth")
        os.makedirs(svfold_corr, exist_ok=True)
        os.makedirs(svfold_depth, exist_ok=True)

        svcorr_root = os.path.join(svfold_corr, "corr_{}_{}.pickle".format(seq.split('/')[1], str(index).zfill(10)))
        with open(svcorr_root, 'wb') as handle:
            pickle.dump(corr_list_rec, handle, protocol=pickle.HIGHEST_PROTOCOL)

        svdepth_root = os.path.join(svfold_depth, "depth_{}_{}.pickle".format(seq.split('/')[1], str(index).zfill(10)))
        with open(svdepth_root, 'wb') as handle:
            pickle.dump(depth_list_rec, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if debug_mode:
            # Some necessary visualizatiron
            tensor2disp(occmask, vmax=1, viewind=0).show()
            tensor2disp(torch.from_numpy(perm_selector).float().unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()
            tensor2disp(torch.from_numpy((occmask_np == 0) * perm_selector).float().unsqueeze(0).unsqueeze(0), vmax=1, viewind=0).show()

            # Check the correctness of the reconstruction
            for kk in rindices:
                if kk == 0:
                    continue
                xxf = xx[perm_selector]
                yyf = yy[perm_selector]
                ck1 = np.array(imgreconrec[kk])[yyf, xxf, :]
                ck2 = imgreconPermrec[kk][keyidx, :, :]
                co_sel = (np.sum(ck1, axis=1) > 0) * (np.sum(ck2, axis=1) > 0)
                if np.sum(co_sel) > 0:
                    ck1 = ck1[co_sel, :]
                    ck2 = ck2[co_sel, :].astype(np.uint8)
                    diff = np.sum(np.abs(ck1 - ck2))
                    assert diff == 0

            rndview = np.random.randint(0, np.sum(perm_selector))
            plt.figure()
            plt.scatter(pts2dperm_v2_batch[:, rndview, 0], pts2dperm_v2_batch[:, rndview, 1])

            rndview = np.random.randint(0, np.sum(perm_selector))
            plt.figure()
            for kk in range(1, idx_range + 1):
                tmp_selector = accum_abserr_rec[kk][:, rndview] > 0
                if np.sum(tmp_selector) > 0:
                    c1, _ = pearsonr(accum_abserr_rec[kk][keyidx::, rndview], rat_range[keyidx::])
                    c2, _ = pearsonr(accum_abserr_rec[kk][0:keyidx + 1, rndview], rat_range[0:keyidx + 1])
                    c = (c1 - c2) / 2
                    print(kk, c)
                    plt.plot(rat_range[tmp_selector], accum_abserr_rec[kk][tmp_selector, rndview])
            plt.legend(['frm1', 'frm2', 'frm3', 'frm4', 'frm5'])

        if dovls:
            imgreconmDs = list()
            for kk in rindices:
                imgreconmDs.append(np.array(imgreconmDrec[kk]))
            imgreconmDs = np.concatenate(imgreconmDs, axis=0)
            imgreconmDsFig = Image.fromarray(imgreconmDs)
            imgreconmDsFig.save(os.path.join(args.vls_root_recon, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))))

            for kk in range(1, idx_range + 1):
                nrgb = 3
                t_rindices = [-kk, 0, kk]
                vls_root_scatter = os.path.join(args.vls_root_scatter, str(kk).zfill(2))
                vls_root_scatterocc = os.path.join(args.vls_root_scatterocc, str(kk).zfill(2))

                os.makedirs(vls_root_scatter, exist_ok=True)
                os.makedirs(vls_root_scatterocc, exist_ok=True)

                cm = plt.get_cmap('magma')
                fig, axs = plt.subplots(nrgb, 1, figsize=(16, 9))

                mDPredFigs = list()
                figmDpred = tensor2disp(1 / torch.from_numpy(mdPred).unsqueeze(0).unsqueeze(0), vmax=0.15, viewind=0)
                figmDpred = np.array(figmDpred)
                for ri in t_rindices:
                    figmDpredc = np.copy(figmDpred)
                    if ri != 0:
                        figmDpredc[occmasks[ri] == 0, :] = 0
                    mDPredFigs.append(figmDpredc)
                mDPredFig = np.concatenate(mDPredFigs, axis=0)
                Image.fromarray(mDPredFig).save(os.path.join(vls_root_scatterocc, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))))

                for idx, k in enumerate(t_rindices):
                    selector = (mdGt > 0) * occmasks[k]
                    colors = cm(1 / mdGt[selector] / 0.15)[:, 0:3]
                    axs[idx].scatter(pts2drec[k][selector, 0], pts2drec[k][selector, 1], 0.5, colors)
                    axs[idx].imshow(imgrec[k])
                    axs[idx].axis('off')

                plt.savefig(os.path.join(vls_root_scatter, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))), bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()
    return

def draw_figure(args, entries):
    nbins = 100
    ebins = np.linspace(np.log(1), np.log(80), nbins)
    ebins = np.exp(ebins)

    corr_list_rect = dict()
    depth_list_rect = dict()
    for kk in range(1, args.idx_range + 1):
        corr_list_rect[kk] = list()
        depth_list_rect[kk] = list()

    for entry in entries:
        seq, index, _ = entry.split(' ')
        svfold_corr = os.path.join(args.tmp_restorage, seq, "corr")
        svfold_depth = os.path.join(args.tmp_restorage, seq, "depth")

        svcorr_root = os.path.join(svfold_corr, "corr_{}_{}.pickle".format(seq.split('/')[1], str(index).zfill(10)))
        svdepth_root = os.path.join(svfold_depth, "depth_{}_{}.pickle".format(seq.split('/')[1], str(index).zfill(10)))
        if not os.path.exists(svcorr_root) or not os.path.exists(svcorr_root):
            continue
        corr_list_rec = pickle.load(open(svcorr_root, "rb"))
        depth_list_rec = pickle.load(open(svdepth_root, "rb"))

        for kk in range(1, args.idx_range + 1):
            corr_list_rect[kk] += corr_list_rec[kk]
            depth_list_rect[kk] += depth_list_rec[kk]


    plt.figure(figsize=(16, 9))
    for kk in range(1, args.idx_range + 1):
        sccx = list()
        sccy = list()
        c_corrarr = np.array(corr_list_rect[kk])
        c_deptharr = np.array(depth_list_rect[kk])
        indices = np.digitize(c_deptharr, ebins)
        for kkk in range(nbins):
            tmpselector = indices == kkk
            if np.sum(indices == kkk) == 0:
                continue
            mean_corr = np.mean(c_corrarr[tmpselector])
            sccx.append(ebins[kkk])
            sccy.append(mean_corr)
        plt.plot(sccx, sccy)
    plt.xlabel('Depth in Meters')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Supervision Signal Quality of %d images with max log diff %f' % (len(entries), args.maxdiff))

    legendtxt = list()
    for kk in range(1, args.idx_range + 1):
        legendtxt.append('{} frames'.format(kk * 2))
    plt.legend(legendtxt)

    plt.savefig(os.path.join(args.vls_root_errcurve, "MaxLogDiff: %f.png" % args.maxdiff), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--odom_root', type=str)
    parser.add_argument('--mdGt_root', type=str)
    parser.add_argument('--mdPred_root', type=str)
    parser.add_argument('--flowPred_root', type=str)
    parser.add_argument('--pose_root', type=str)
    parser.add_argument('--vls_root_scatter', type=str)
    parser.add_argument('--vls_root_scatterocc', type=str)
    parser.add_argument('--vls_root_recon', type=str)
    parser.add_argument('--vls_root_errcurve', type=str)
    parser.add_argument('--tmp_restorage', type=str)
    parser.add_argument('--nprocs', type=int, default=5)
    parser.add_argument('--maxdiff', type=float, default=0.3)
    parser.add_argument('--idx_range', type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(2021)
    np.random.seed(2021)
    random.seed(2021)

    args.tmp_restorage = os.path.join(args.tmp_restorage, "maxdiff: %f" % (args.maxdiff))
    os.makedirs(args.tmp_restorage, exist_ok=True)

    os.makedirs(args.vls_root_scatter, exist_ok=True)
    os.makedirs(args.vls_root_scatterocc, exist_ok=True)
    os.makedirs(args.vls_root_recon, exist_ok=True)
    os.makedirs(args.vls_root_errcurve, exist_ok=True)

    num_frames = 1000

    entries = get_all_entries(args)
    random.shuffle(entries)
    mp.spawn(calc_err_curve_tmpre, nprocs=args.nprocs, args=(args, entries[0:num_frames], False))

    draw_figure(args, entries)