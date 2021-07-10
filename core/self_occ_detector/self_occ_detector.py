import torch
import selfoccdtc_ops
import torch.nn as nn
import numpy as np
import time

class self_occ_detector(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(self_occ_detector, self).__init__()

    @staticmethod
    def forward(ctx, intrinsic, pose, depthmap, minoc_dist):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        assert intrinsic.dtype == torch.float32 and pose.dtype == torch.float32 and depthmap.dtype == torch.float32, print("self occ detector input data type error")
        assert intrinsic.ndim == 3 and pose.ndim == 3 and depthmap.ndim == 4, print("self occ detector input shape error")

        bz, _, h, w = depthmap.shape

        minsr_dist = 1
        pose_pr = torch.clone(pose)
        pose_pr[:, 0:3, 3] = 0

        cam_org_3d = torch.zeros([bz, 4, 1], device=depthmap.device, dtype=torch.float32)
        cam_org_3d[:, 3, :] = 1
        epp1 = intrinsic @ torch.inverse(pose) @ cam_org_3d  # Epipole on frame 1, projection of camera 2
        epp1[:, 0, 0] = epp1[:, 0, 0] / epp1[:, 2, 0]
        epp1[:, 1, 0] = epp1[:, 1, 0] / epp1[:, 2, 0]
        epp1 = epp1[:, 0:2, 0]
        epp2 = intrinsic @ pose @ cam_org_3d  # Epipole on frame 2, projection of camera 1
        epp2[:, 0, 0] = epp2[:, 0, 0] / epp2[:, 2, 0]
        epp2[:, 1, 0] = epp2[:, 1, 0] / epp2[:, 2, 0]
        epp2 = epp2[:, 0:2, 0]

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        xx = torch.from_numpy(xx).view([1, h, w, 1, 1]).expand([bz, -1, -1, -1, -1]).float().cuda(depthmap.device)
        yy = torch.from_numpy(yy).view([1, h, w, 1, 1]).expand([bz, -1, -1, -1, -1]).float().cuda(depthmap.device)
        ones = torch.ones_like(xx)
        depthmap_rs = depthmap.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        pts3d_v1_batch = torch.cat([xx * depthmap_rs, yy * depthmap_rs, depthmap_rs, ones], axis=3)
        pts2d_v1_batch = torch.cat([xx, yy], axis=3).squeeze(-1)

        pM1 = intrinsic @ pose @ torch.inverse(intrinsic)
        pM1 = pM1.view([bz, 1, 1, 4, 4]).expand([-1, h, w, -1, -1])
        pts3d_v2_batch = pM1 @ pts3d_v1_batch
        pts2d_v2_batch = torch.clone(pts3d_v2_batch)
        pts2d_v2_batch[:, :, :, 0, 0] = pts2d_v2_batch[:, :, :, 0, 0] / pts2d_v2_batch[:, :, :, 2, 0]
        pts2d_v2_batch[:, :, :, 1, 0] = pts2d_v2_batch[:, :, :, 1, 0] / pts2d_v2_batch[:, :, :, 2, 0]
        pts2d_v2_batch = pts2d_v2_batch[:, :, :, 0:2, 0]
        out_range_selecor = (pts3d_v2_batch[:, :, :, 2, 0] >= 0) * (pts2d_v2_batch[:, :, :, 0] >= 0) * (pts2d_v2_batch[:, :, :, 0] <= w - 1) * (pts2d_v2_batch[:, :, :, 1] >= 0) * (pts2d_v2_batch[:, :, :, 1] <= h - 1)
        out_range_selecor = (out_range_selecor == 0)
        occ_selector = torch.clone(out_range_selecor).unsqueeze(1)

        pM2 = intrinsic @ torch.inverse(pose_pr) @ torch.inverse(intrinsic) @ intrinsic @ pose @ torch.inverse(intrinsic)
        pM2 = pM2.view([bz, 1, 1, 4, 4]).expand([-1, h, w, -1, -1])
        pts2dsrch_v1_batch = pM2 @ pts3d_v1_batch
        pts2dsrch_v1_batch[:, :, :, 0, 0] = pts2dsrch_v1_batch[:, :, :, 0, 0] / pts2dsrch_v1_batch[:, :, :, 2, 0]
        pts2dsrch_v1_batch[:, :, :, 1, 0] = pts2dsrch_v1_batch[:, :, :, 1, 0] / pts2dsrch_v1_batch[:, :, :, 2, 0]
        pts2dsrch_v1_batch = pts2dsrch_v1_batch[:, :, :, 0:2, 0]

        srh_distance = torch.sqrt(torch.sum((pts2dsrch_v1_batch - pts2d_v1_batch) ** 2 + 1e-10, axis=3))

        epp1 = epp1.contiguous().float()
        epp2 = epp2.contiguous().float()
        pts2dsrch_v1_batch = pts2dsrch_v1_batch.contiguous().float()
        pts2d_v1_batch = pts2d_v1_batch.contiguous().float()
        pts2d_v2_batch = pts2d_v2_batch.contiguous().float()
        srh_distance = srh_distance.contiguous().float()
        occ_selector = occ_selector.contiguous().int()
        minsr_dist = float(minsr_dist)
        minoc_dist = float(minoc_dist)
        bz = int(bz)
        h = int(h)
        w = int(w)
        selfoccdtc_ops.self_occ_dtc(epp1, epp2, pts2dsrch_v1_batch, pts2d_v1_batch, pts2d_v2_batch, srh_distance, occ_selector, minsr_dist, minoc_dist, bz, h, w)
        return occ_selector == 1

    @staticmethod
    def forward_debug(ctx, intrinsic, pose, depthmap, minoc_dist, ckz, cky, ckx, silent):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        assert intrinsic.dtype == torch.float32 and pose.dtype == torch.float32 and depthmap.dtype == torch.float32, print("self occ detector input data type error")
        assert intrinsic.ndim == 3 and pose.ndim == 3 and depthmap.ndim == 4, print("self occ detector input shape error")

        bz, _, h, w = depthmap.shape

        minsr_dist = 1
        pose_pr = torch.clone(pose)
        pose_pr[:, 0:3, 3] = 0

        cam_org_3d = torch.zeros([bz, 4, 1], device=depthmap.device, dtype=torch.float32)
        cam_org_3d[:, 3, :] = 1
        epp1 = intrinsic @ torch.inverse(pose) @ cam_org_3d  # Epipole on frame 1, projection of camera 2
        epp1[:, 0, 0] = epp1[:, 0, 0] / epp1[:, 2, 0]
        epp1[:, 1, 0] = epp1[:, 1, 0] / epp1[:, 2, 0]
        epp1 = epp1[:, 0:2, 0]
        epp2 = intrinsic @ pose @ cam_org_3d  # Epipole on frame 2, projection of camera 1
        epp2[:, 0, 0] = epp2[:, 0, 0] / epp2[:, 2, 0]
        epp2[:, 1, 0] = epp2[:, 1, 0] / epp2[:, 2, 0]
        epp2 = epp2[:, 0:2, 0]

        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        xx = torch.from_numpy(xx).view([1, h, w, 1, 1]).expand([bz, -1, -1, -1, -1]).float().cuda(depthmap.device)
        yy = torch.from_numpy(yy).view([1, h, w, 1, 1]).expand([bz, -1, -1, -1, -1]).float().cuda(depthmap.device)
        ones = torch.ones_like(xx)
        depthmap_rs = depthmap.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        pts3d_v1_batch = torch.cat([xx * depthmap_rs, yy * depthmap_rs, depthmap_rs, ones], axis=3)
        pts2d_v1_batch = torch.cat([xx, yy], axis=3).squeeze(-1)

        pM1 = intrinsic @ pose @ torch.inverse(intrinsic)
        pM1 = pM1.view([bz, 1, 1, 4, 4]).expand([-1, h, w, -1, -1])
        pts3d_v2_batch = pM1 @ pts3d_v1_batch
        pts2d_v2_batch = torch.clone(pts3d_v2_batch)
        pts2d_v2_batch[:, :, :, 0, 0] = pts2d_v2_batch[:, :, :, 0, 0] / pts2d_v2_batch[:, :, :, 2, 0]
        pts2d_v2_batch[:, :, :, 1, 0] = pts2d_v2_batch[:, :, :, 1, 0] / pts2d_v2_batch[:, :, :, 2, 0]
        pts2d_v2_batch = pts2d_v2_batch[:, :, :, 0:2, 0]
        out_range_selecor = (pts3d_v2_batch[:, :, :, 2, 0] >= 0) * (pts2d_v2_batch[:, :, :, 0] >= 0) * (pts2d_v2_batch[:, :, :, 0] <= w - 1) * (pts2d_v2_batch[:, :, :, 1] >= 0) * (pts2d_v2_batch[:, :, :, 1] <= h - 1)
        out_range_selecor = (out_range_selecor == 0)
        occ_selector = torch.clone(out_range_selecor).unsqueeze(1)

        pM2 = intrinsic @ torch.inverse(pose_pr) @ torch.inverse(intrinsic) @ intrinsic @ pose @ torch.inverse(intrinsic)
        pM2 = pM2.view([bz, 1, 1, 4, 4]).expand([-1, h, w, -1, -1])
        pts2dsrch_v1_batch = pM2 @ pts3d_v1_batch
        pts2dsrch_v1_batch[:, :, :, 0, 0] = pts2dsrch_v1_batch[:, :, :, 0, 0] / pts2dsrch_v1_batch[:, :, :, 2, 0]
        pts2dsrch_v1_batch[:, :, :, 1, 0] = pts2dsrch_v1_batch[:, :, :, 1, 0] / pts2dsrch_v1_batch[:, :, :, 2, 0]
        pts2dsrch_v1_batch = pts2dsrch_v1_batch[:, :, :, 0:2, 0]

        seach_distance = torch.sqrt(torch.sum((pts2dsrch_v1_batch - pts2d_v1_batch) ** 2 + 1e-10, axis=3))
        if not silent:
            print(seach_distance[ckz, cky, ckx], epp1[ckz], epp2[ckz], torch.sum(occ_selector[ckz]))
        return