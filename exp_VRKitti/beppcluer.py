import numpy as np
import torch
import os
import PIL.Image as Image
import cv2
from core.utils.utils import tensor2disp
from core.utils.frame_utils import readFlowVRKitti

def read_splits(project_rootdir):
    split_root = os.path.join(project_rootdir, 'exp_VRKitti', 'splits')
    train_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'training_split.txt'), 'r')]
    evaluation_entries = [x.rstrip('\n') for x in open(os.path.join(split_root, 'evaluation_split.txt'), 'r')]
    return train_entries, evaluation_entries

def read_move_info(extrinisic_path, intrinsic_path, objpose_path):
    intrinsic_entries = [x.rstrip('\n') for x in open(intrinsic_path, 'r')]
    intrinsic_dict = dict()
    for entry in intrinsic_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = horarr[0]
        camidx = horarr[1]
        intrinsic_key = "{}_{}".format(str(frmidx).zfill(5), camidx)

        intrinsic = np.eye(4)
        intrinsic[0, 0] = horarr[2]
        intrinsic[1, 1] = horarr[3]
        intrinsic[0, 2] = horarr[4]
        intrinsic[1, 2] = horarr[5]
        intrinsic_dict[intrinsic_key] = intrinsic

    extrisnic_entries = [x.rstrip('\n') for x in open(extrinisic_path, 'r')]
    extrinsic_dict = dict()
    for entry in extrisnic_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = horarr[0]
        camidx = horarr[1]
        extrinsic_key = "{}_{}".format(str(frmidx).zfill(5), camidx)
        extrinsic_dict[extrinsic_key] = np.reshape(np.array([horarr[2::]]), [4,4])

    objpose_entries = [x.rstrip('\n') for x in open(objpose_path, 'r')]
    objpose_dict = dict()
    for entry in objpose_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = horarr[0]
        camidx = horarr[1]
        trackidx = horarr[2]
        objpose_key = "{}_{}_{}".format(str(frmidx).zfill(5), camidx, str(trackidx).zfill(5))

        objpose_dict[objpose_key] = np.array(horarr[3::])

    return intrinsic_dict, extrinsic_dict, objpose_dict


def read_move_info1(extrinisic_path, intrinsic_path, objpose_path, scene_name):
    intrinsic_entries = [x.rstrip('\n') for x in open(intrinsic_path, 'r')]
    intrinsic_dict = dict()
    for entry in intrinsic_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = int(horarr[0])
        camidx = int(horarr[1])
        intrinsic_key = "{}_frm{}_cam{}".format(scene_name, str(frmidx).zfill(5), camidx)

        intrinsic = np.eye(4)
        intrinsic[0, 0] = horarr[2]
        intrinsic[1, 1] = horarr[3]
        intrinsic[0, 2] = horarr[4]
        intrinsic[1, 2] = horarr[5]
        intrinsic_dict[intrinsic_key] = intrinsic

    extrisnic_entries = [x.rstrip('\n') for x in open(extrinisic_path, 'r')]
    extrinsic_dict = dict()
    for entry in extrisnic_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = int(horarr[0])
        camidx = int(horarr[1])
        extrinsic_key = "{}_frm{}_cam{}".format(scene_name, str(frmidx).zfill(5), camidx)
        extrinsic_dict[extrinsic_key] = np.reshape(np.array([horarr[2::]]), [4,4])

    objpose_entries = [x.rstrip('\n') for x in open(objpose_path, 'r')]
    objpose_dict = dict()
    for entry in objpose_entries[1::]:
        horarr = np.array(list(map(float, entry.split(' '))))
        frmidx = int(horarr[0])
        camidx = int(horarr[1])
        trackidx = int(horarr[2])
        objpose_key = "{}_frm{}_cam{}_obj{}".format(scene_name, str(frmidx).zfill(5), camidx, str(trackidx).zfill(5))

        objpose_dict[objpose_key] = np.array(horarr[3::])

    return intrinsic_dict, extrinsic_dict, objpose_dict

if __name__ == '__main__':
    vrkittiroot = '/media/shengjie/disk1/data/virtual_kitti_organized'
    project_rootdir = '/home/shengjie/Documents/supporting_projects/RAFT'

    train_entries, evaluation_entries = read_splits(project_rootdir)
    tids = np.random.randint(0, len(train_entries), 2)

    img1s = list()
    img2s = list()

    depthmaps = list()
    flowmaps = list()
    instancemaps = list()
    intrinsics = list()
    extrinsics = list()
    mvinfo = list()
    for tid in tids:
        entry = train_entries[tid]
        sceneidx, envn, frmidx = entry.split(' ')
        frmidx = int(frmidx)

        img1s.append(Image.open(os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(frmidx).zfill(5)))))
        img2s.append(Image.open(os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'rgb', 'Camera_0', "rgb_{}.jpg".format(str(frmidx + 1).zfill(5)))))

        depthmap = cv2.imread(os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'depth', 'Camera_0', "depth_{}.png".format(str(frmidx).zfill(5))), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depthmap = depthmap / 100
        depthmaps.append(depthmap)

        flowmap = readFlowVRKitti(os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'forwardFlow', 'Camera_0', "flow_{}.png".format(str(frmidx).zfill(5))))
        flowmaps.append(flowmap)

        instancemap = cv2.imread(os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'frames', 'instanceSegmentation', 'Camera_0', "instancegt_{}.png".format(str(frmidx).zfill(5))), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        instancemap = instancemap[:, :, 0]
        instancemap = np.array(instancemap).astype(np.int32) - 1
        instancemaps.append(instancemap)

        extrinisic_path = os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'extrinsic.txt')
        intrinsic_path = os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'intrinsic.txt')
        objpose_path = os.path.join(vrkittiroot, "Scene{}".format(sceneidx.zfill(2)), envn, 'pose.txt')
        # intrinsic_dict, extrinsic_dict, objpose_dict = read_move_info1(extrinisic_path, intrinsic_path, objpose_path, "Scene{}".format(sceneidx.zfill(2)))
        intrinsic_dict, extrinsic_dict, objpose_dict = read_move_info(extrinisic_path, intrinsic_path, objpose_path)

        tensor2disp(torch.from_numpy(1 / depthmap).unsqueeze(0).unsqueeze(0), vmax=0.15, viewind=0).show()
        tensor2disp(torch.from_numpy(instancemap).unsqueeze(0).unsqueeze(0) == 0, vmax=0.15, viewind=0).show()