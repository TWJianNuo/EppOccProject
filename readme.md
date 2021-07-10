# Epipolar Occlusion Detection Project
We introduce an algorithm to do occlusion detection using an off-the-shelf algorithm. 
Also, we demonstrate the importance by showing its benefits on a few downstream tasks.

## Run on synthetic data
* First, prepare VRKITTI2 dataset by downloading from [VRKITTI2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/). You need to at least include those packages:
    * vkitti_2.0.3_rgb.tar
    * vkitti_2.0.3_depth.tar
    * vkitti_2.0.3_classSegmentation.tar
    * vkitti_2.0.3_instanceSegmentation.tar
    * vkitti_2.0.3_textgt.tar.gz
    * vkitti_2.0.3_forwardFlow.tar
    * vkitti_2.0.3_backwardFlow.tar

* Extract all rar into a common folder and run:
    ```
    python exp_VRKITTI/organize_VRKitti2.py --virtualkitti_root your_vrkitti2_root --export_root your_target_location_for_organized_vrkitti2
    ```
* Finally run the following script to get a visualization of algorithm:
    ```
    python exp_VRKITTI/dev/self_epp_occ_numba_forward.py --vrkittiroot your_target_location_for_organized_vrkitti2 --vlsroot your_target_location_for_visualization
    ```