# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import PIL.Image as Image

import json
from exp_KITTI_oraclepose.layers import *

from exp_KITTI_oraclepose.dataset_kitti_eigen import KITTI_eigen
import exp_KITTI_oraclepose.networks as networks
from core.utils.utils import readlines, tensor2disp, tensor2rgb, tensor2grad, sec_to_hm_str
import argparse
from core import self_occ_detector
import tqdm
import sys
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

file_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # the directory that options.py resides in

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, prj_root)
sys.path.append('core')
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

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        self.use_pose_net = False

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        fpath = os.path.join(file_dir, 'exp_KITTI_oraclepose', "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("test"))

        train_filenames = train_filenames
        val_filenames = val_filenames

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = KITTI_eigen(entries=train_filenames, dataset_root=self.opt.dataset_root, depthgt_root=self.opt.depthgt_root,
                                    RANSACPose_root=self.opt.RANSACPose_root, inheight=self.opt.height, inwidth=self.opt.width, inDualDirFrames=args.inDualDirFrames, istrain=True, muteaug=False, isgarg=True)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = KITTI_eigen(entries=val_filenames, dataset_root=self.opt.dataset_root, depthgt_root=self.opt.depthgt_root,
                                    RANSACPose_root=self.opt.RANSACPose_root, inheight=320, inwidth=1216, inDualDirFrames=0, istrain=False, muteaug=True, isgarg=True)
        self.val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.reconimg = ReconImages()
        self.sod = self_occ_detector.apply

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.maxa1 = -1e10
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["total_loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if np.mod(self.step, 100) == 0:
                self.log_time(batch_idx, duration, losses["total_loss"].cpu().data)

                if "depthgt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)

            if np.mod(self.step, 5000) == 0:
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if key == 'tag':
                continue
            elif key == 'imgr' or key == 'poser':
                for k, kipt in inputs[key].items():
                    inputs[key][k] = kipt.to(self.device)
            else:
                inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs['inrgb_augmented'])
        outputs = self.models["depth"](features)

        outputs = self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        totcount = 0
        eval_measures_depth_nps = list()
        losses = dict()
        self.set_eval()
        for _, inputs in enumerate(tqdm.tqdm(self.val_loader)):
            with torch.no_grad():
                for key, ipt in inputs.items():
                    if key == 'tag':
                        continue
                    elif key == 'imgr' or key == 'poser':
                        for k, kipt in inputs[key].items():
                            inputs[key][k] = kipt.to(self.device)
                    else:
                        inputs[key] = ipt.to(self.device)

                features = self.models["encoder"](inputs['inrgb_augmented'])
                outputs = self.models["depth"](features)

                disp = outputs[("disp", 0)]
                _, depth_pred = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

                depthgt = inputs['depthgt']
                selector = (depthgt > 0).float()
                depth_gt_flatten = depthgt[selector == 1].cpu().numpy()
                pred_depth_flatten = depth_pred[selector == 1].cpu().numpy()

                pred_depth_flatten *= np.mean(depth_gt_flatten) / np.mean(pred_depth_flatten)

                eval_measures_depth_np = compute_errors(gt=depth_gt_flatten, pred=pred_depth_flatten)
                eval_measures_depth_nps.append(eval_measures_depth_np)
                totcount += 1

        eval_measures_depth_nps = np.stack(eval_measures_depth_nps, axis=0)
        eval_measures_depth_nps = np.sum(eval_measures_depth_nps, axis=0) / totcount

        print('Computing Depth errors for %f eval samples' % (totcount))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_depth_nps[i]), end='')
        print('{:7.3f}'.format(eval_measures_depth_nps[8]))

        eval_measures_name = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
        for i in range(9):
            losses[eval_measures_name[i]] = eval_measures_depth_nps[i]

        if self.maxa1 < losses['d1']:
            self.maxa1 = losses['d1']
            self.best_measure = losses
            self.save_model('besta1')

        print('Best Performance:')
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
        for k, name in enumerate(eval_measures_name):
            if k < 8:
                print('{:7.3f}, '.format(self.best_measure[name]), end='')
            else:
                print('{:7.3f}'.format(self.best_measure[name]))

        self.log('val', None, None, losses)
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", scale)] = depth

        # Compute Reprojection loss
        occmaskrec = dict()
        for k in inputs['imgr'].keys():
            if k == 0:
                continue
            occmaskrec[k] = (self.sod(inputs['intrinsic'], inputs['poser'][k], outputs[("depth", 0)], float(1e10)) == 0).float()
        outputs['occmaskrec'] = occmaskrec

        # Compute Reconstruction image
        for scale in self.opt.scales:
            outputs['reconImgs', scale] = self.reconimg(inputs['imgr'], outputs[("depth", scale)], inputs['intrinsic'], inputs['poser'])

        return outputs

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            closs = torch.zeros_like(outputs[('depth', 0)])
            ccount = torch.zeros_like(outputs[('depth', 0)])
            for k in inputs['imgr'].keys():
                if k == 0:
                    continue
                closs += self.compute_reprojection_loss(pred=outputs['reconImgs', scale][k], target=inputs['imgr'][0]) * outputs['occmaskrec'][k]
                ccount += outputs['occmaskrec'][k]
            avecloss = closs / (ccount + 1e-5)

            if scale == 0:
                outputs['avecloss'] = avecloss

            losses['closs', scale] = torch.sum(avecloss) / (torch.sum(avecloss > 0) + 1)
            total_loss += losses['closs', scale]

        total_loss = total_loss / len(self.opt.scales)
        losses['total_loss'] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        bz, _, h, w = inputs["depthgt"].shape
        depth_pred = outputs[("depth", 0)].detach()

        depth_gt = inputs["depthgt"]
        mask = depth_gt > 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        if mode == 'val':
            return

        imgreconc = list()
        for k in range(-self.opt.inDualDirFrames, self.opt.inDualDirFrames + 1):
            if k == 0:
                imgreconc.append(np.array(tensor2rgb(inputs['imgr'][k], viewind=0)))
            else:
                imgrecon = np.array(tensor2rgb(outputs['reconImgs', 0][k], viewind=0))
                imgocc = outputs['occmaskrec'][k][0].squeeze().cpu().numpy()
                imgrecon = imgrecon * np.expand_dims(imgocc, axis=2)
                imgrecon = imgrecon.astype(np.uint8)
                imgreconc.append(imgrecon)
        imgreconc.append(np.array(tensor2disp(1 / outputs[('depth', 0)], percentile=95, viewind=0)))
        imgreconc.append(np.array(tensor2disp(outputs['avecloss'], vmax=0.5, viewind=0)))
        imgreconc = np.concatenate(imgreconc, axis=0)
        writer.add_image('"img_recon"', (torch.from_numpy(imgreconc).float() / 255).permute([2, 0, 1]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, name=None):
        """Save model weights to disk
        """
        if name is None:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        else:
            save_folder = os.path.join(self.log_path, "models", name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        print("Model saved to %s" % save_folder)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # PATHS
    parser.add_argument("--dataset_root",
                             type=str)
    parser.add_argument("--depthgt_root",
                             type=str)
    parser.add_argument("--RANSACPose_root",
                             type=str)
    parser.add_argument("--log_dir",
                             type=str,
                             help="log directory",
                             default=os.path.join(os.path.expanduser("~"), "tmp"))

    # TRAINING options
    parser.add_argument("--inDualDirFrames",
                             type=int
                        )
    parser.add_argument("--model_name",
                             type=str,
                             help="the name of the folder to save the model in",
                             default="mdp")
    parser.add_argument("--split",
                             type=str,
                             help="which training split to use",
                             choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                             default="eigen_zhou")
    parser.add_argument("--num_layers",
                             type=int,
                             help="number of resnet layers",
                             default=18,
                             choices=[18, 34, 50, 101, 152])
    parser.add_argument("--height",
                             type=int,
                             help="input image height",
                             default=320)
    parser.add_argument("--width",
                             type=int,
                             help="input image width",
                             default=1024)
    parser.add_argument('--evalheight', type=int, default=320)
    parser.add_argument('--evalwidth', type=int, default=1216)
    parser.add_argument("--disparity_smoothness",
                             type=float,
                             help="disparity smoothness weight",
                             default=1e-3)
    parser.add_argument("--scales",
                             nargs="+",
                             type=int,
                             help="scales used in the loss",
                             default=[0, 1, 2, 3])
    parser.add_argument("--min_depth",
                             type=float,
                             help="minimum depth",
                             default=0.1)
    parser.add_argument("--max_depth",
                             type=float,
                             help="maximum depth",
                             default=100.0)
    parser.add_argument("--use_stereo",
                             help="if set, uses stereo pair for training",
                             action="store_true")
    parser.add_argument("--frame_ids",
                             nargs="+",
                             type=int,
                             help="frames to load",
                             default=[0, -1, 1])

    # OPTIMIZATION options
    parser.add_argument("--batch_size",
                             type=int,
                             help="batch size",
                             default=12)
    parser.add_argument("--learning_rate",
                             type=float,
                             help="learning rate",
                             default=1e-4)
    parser.add_argument("--num_epochs",
                             type=int,
                             help="number of epochs",
                             default=25)
    parser.add_argument("--scheduler_step_size",
                             type=int,
                             help="step size of the scheduler",
                             default=15)
    parser.add_argument("--save_frequency",
                             type=int,
                             default=1)


    # ABLATION options
    parser.add_argument("--avg_reprojection",
                             help="if set, uses average reprojection loss",
                             action="store_true")
    parser.add_argument("--disable_automasking",
                             help="if set, doesn't do auto-masking",
                             action="store_true")
    parser.add_argument("--weights_init",
                             type=str,
                             help="pretrained or scratch",
                             default="pretrained",
                             choices=["pretrained", "scratch"])
    parser.add_argument("--pose_model_input",
                             type=str,
                             help="how many images the pose network gets",
                             default="pairs",
                             choices=["pairs", "all"])
    parser.add_argument("--pose_model_type",
                             type=str,
                             help="normal or shared",
                             default="separate_resnet",
                             choices=["posecnn", "separate_resnet", "shared"])

    # SYSTEM options
    parser.add_argument("--num_workers",
                             type=int,
                             help="number of dataloader workers",
                             default=12)

    # LOADING options
    parser.add_argument("--load_weights_folder",
                             type=str,
                             help="name of model to load")
    parser.add_argument("--models_to_load",
                             nargs="+",
                             type=str,
                             help="models to load",
                             default=["encoder", "depth"])

    # LOGGING options
    parser.add_argument("--log_frequency",
                             type=int,
                             help="number of batches between each tensorboard log",
                             default=250)

    # EVALUATION options
    parser.add_argument("--post_process",
                             help="if set will perform the flipping post processing "
                                  "from the original monodepth paper",
                             action="store_true")

    args = parser.parse_args()
    args.log_dir = os.path.join(file_dir, args.log_dir)

    trainer = Trainer(args)
    trainer.train()