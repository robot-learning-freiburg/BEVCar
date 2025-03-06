# BEVCar
[**arXiv**](https://arxiv.org/abs/2403.11761) | [**IEEE Xplore**](https://ieeexplore.ieee.org/document/10802147) | [**Website**](http://bevcar.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/bB_k_6IvPHQ?feature=shared)

This repository is the official implementation of the paper:

> **BEVCar: Camera-Radar Fusion for BEV Map and Object Segmentation**
>
> [Jonas Schramm]()&ast;, [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/)&ast;, [B Ravi Kiran](), [Senthil Yogamani](), [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/), and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada). <br>
> &ast;Equal contribution. <br>
>
> *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2024, pp. 1435-1442 <br>
> **IROS Best Paper on Cognitive Robotics - Winner** <br>
> **IROS Best Student Paper - Finalist**

<p align="center">
  <img src="./assets/bevcar_overview.png" alt="Overview of BEVCar approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@inproceedings{schramm2024bevcar,
  author={Schramm, Jonas and Vödisch, Niclas and Petek, Kürsat and Kiran, B Ravi and Yogamani, Senthil and Burgard, Wolfram and Valada, Abhinav},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={{BEVCar}: Camera-Radar Fusion for BEV Map and Object Segmentation}, 
  year={2024},
  pages={1435-1442},
}
```


## 📔 Abstract

Semantic scene segmentation from a bird's-eye-view (BEV) perspective plays a crucial role in facilitating planning and decision-making for mobile robots. Although recent vision-only methods have demonstrated notable advancements in performance, they often struggle under adverse illumination conditions such as rain or nighttime. While active sensors offer a solution to this challenge, the prohibitively high cost of LiDARs remains a limiting factor. Fusing camera data with automotive radars poses a more inexpensive alternative but has received less attention in prior research. In this work, we aim to advance this promising avenue by introducing BEVCar, a novel approach for joint BEV object and map segmentation. The core novelty of our approach lies in first learning a point-based encoding of raw radar data, which is then leveraged to efficiently initialize the lifting of image features into the BEV space. We perform extensive experiments on the nuScenes dataset and demonstrate that BEVCar outperforms the current state of the art. Moreover, we show that incorporating radar information significantly enhances robustness in challenging environmental conditions and improves segmentation performance for distant objects.

## 💾 Data Preparation

We show results on the nuScenes dataset. The nuScenes dataset is available for download on their [website](https://www.nuscenes.org/download).
After downloading, extract the archives to e.g. `DIR: /datasets/nuscenes` resulting in the following folder structure:
```
/datasets/nuscenes - DIR
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	-	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-trainval:	JSON tables that include the meta data and annotations for the train and eval set
    v1.0-test:		JSON tables that include the meta data and annotations for the test set
    v1.0-mini:		JSON tables that include the meta data and annotations for the mini set
```

In order for our code to find all nuScenes data, please reference your nuScenes extraction folder (DIR) with the
`data_dir` argument in the [config files](./configs).

By specifying the parameter custom_dataroot and setting `use_pre_scaled_imgs: true` in the [config files](./configs), you can load pre-scaled images to reduce the amount
of data per loaded sample. To create such prescaled images, please refer to the provided
[image_converter](./nuscenes_image_converter.py).

Further details can be found here:
* [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
* [nuScenes tutorials](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/tutorials)

### DAY / RAIN / NIGHT split for the nuScenes val-set
Besides the code for BEVCar we release a custom split from the nuScenes val-set which allows separate inference for three different scene conditions. Namely, we categorized all val-scenes to either be a _DAY_, _RAIN_ or _NIGHT_ scene. Hereby, we select rainy conditions at night to the _NIGHT_ category.

In order to use our custom split for your evaluation, a few changes to the dataloader are necessary. Replace the nuScenes function `create_splits_scenes()[split]` with the provided function `create_drn_eval_split_scenes()[split]` implemented in [`custom_nuscenes_splits.py`](./custom_nuscenes_splits.py).

After that you can get all ordered samples with the function `get_ordered_drn_samples()` that is defined in [`nuscenesdataset.py`](./nuscenes_data.py).

An example for handling the reordered val-set is provided in the [`eval_nuscenes.py`](./eval.py) script.
Note that in order to use this version of the val split, the parameter `do_drn_val_split` must be set to `True` in the config file, which is the case in our provided eval and visualization config files.

---

Please be advised that the following Shapely warnings are expected:
```
../lib/python3.10/site-packages/nuscenes/map_expansion/map_api.py:1823: ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.
../lib/python3.10/site-packages/nuscenes/map_expansion/map_api.py:1824: ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.
../lib/python3.10/site-packages/nuscenes/map_expansion/map_api.py:1838: ShapelyDeprecationWarning: Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.
  for line in lines:
```
- Shapely must be <2.0 since a newer version is not compatible with the nuScenes map API.
- This is handled by installing a suitable Shapely version in the [requirements.txt](./requirements.txt).
- Additionally, we filter them out with: `warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning, module="nuscenes.map_expansion.map_api")`


## 👩‍💻 Code

The following section describes the setup of the environment as well as running the code. This repo includes saved
model checkpoints for various experiments conducted in the released paper.

Please note, that we cannot guarantee a working environment for a deviating setup.

**Prerequisites:**
- conda (miniconda)
- wandb account for logging during training
- nuScenes dataset in the default folder structure as advised [above](#-data-preparation)
- GPU with at least 32GB of memory (for training)
- CUDA version 11.8

### Setup environment and install requirements

```
conda create --name bevcar python=3.10
source activate bevcar
conda install pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pip
conda install xformers -c xformers
pip install -r requirements.txt
```

### Compile deformable attention operators
Further details: [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
```
cd ./nets/ops
sh ./make.sh
# test build
python test.py
```

### Pre-trained network weights
We provide the weights for the following setups:
- BEVCar (default, DINOv2): https://bevcar.cs.uni-freiburg.de/downloads/BEVCar_model-000050000.pth
- BEVCar (ResNet-101): https://bevcar.cs.uni-freiburg.de/downloads/BEVCar_ResNet_model-000050000.pth
- BEVCar (camera-only, DINOv2): https://bevcar.cs.uni-freiburg.de/downloads/CAM_ONLY_model-000050000.pth

After downloading, save the files as:
```
./model_checkpoints/
    BEVCar / model-000050000.pth
    BEVCar_ResNet / model-000050000.pth
    CAM_ONLY / model-000050000.pth
```


### Training
There are two training scripts provided. One utilizing the DataParallel (DP) class of pytorch and one based on the DistributedDataParallel (DDP) class. Furthermore, you  can find respective config files for our experiments in the configs directory.

**Starting the DP training (beneficial for debugging):**
- For changing the number of GPUs adapt the list `device_ids` accordingly
```
CUDA_VISIBLE_DEVICES=0 python train.py --config='configs/train/train_bevcar.yaml'
```

**Starting the DDP training:**
```
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB__SERVICE_WAIT=300 torchrun --nproc_per_node=8 --nnodes=1 --master_port=1234 train_DDP.py --config='configs/train/train_bevcar.yaml'
```
- Replace the config file for the respective experiment or define your own config.
- If you change the parameter `exp_name` in the config files of the experiment a new folder in the [model_checkpoints](./model_checkpoints) directory is created where you can find your model weights for evaluation.
- We trained our BEVCar model on 8 GPUs utilizing roughly 28GB each with a batch size of 1 per GPU and 5 gradient accumulation steps, resulting in an effective batch size of 40.
- If your number of GPUs deviates from our preset, please change the argument `--nproc_per_node` to the number of GPUs and address the respective GPUs with the global variable `CUDA_VISIBLE_DEVICES`.
- The argument `--nnodes=1` should remain unchanged, unless you aim for a parallel training across multiple servers.

### Evaluation
Per default, we evaluate on our DAY/RAIN/NIGHT split and provide metrics for each scenario as well as the overall
metrics on the val split. For the vehicle/object segmentation we calculate metrics for the whole BEV area as well as
in three areas of different distance around the ego-car. Namely, 0-20m, 20-35m and 35m-50m.
After the evaluation of every val sample, the final results are presented in the terminal.

The directory [model_checkpoints](./model_checkpoints) holds the weights of experiments conducted in the paper and are already referenced in their corresponding eval-config files.
To evaluate the proposed BEVCar model, please run the following command:
```
CUDA_VISIBLE_DEVICES=0 python eval.py  --config='configs/eval/eval_bevcar.yaml'
```
- Note, that the eval run supports only a batch size of 1 on a single GPU.

### Visualization
We provide a script ([vis_eval.py](./vis_eval.py)) that generates and stores the model output of
BEVCar while collecting all relevant metrics. For every scene of the val split it creates a folder holding all input
data in form of all 6 camera views and the voxelized radar point cloud in BEV around the ego-car as well as the GT and
predicted segmentations for the respective map classes and the vehicle object class. Additionally, this folder holds a
.txt file with the metrics for every sample as well as the whole scene.
The final eval results for the whole val split can be found in the  terminal as explained in the
[evaluation section](#evaluation).

As an example, you can find the results of the first scene in the
[vis_eval](./vis_eval/50000_BEVCar_VIS_EVAL_scene_001_example) directory.

To run the visualization together with the evaluation, please run the following command on a single GPU:
```
CUDA_VISIBLE_DEVICES=0 python vis_eval.py --config='configs/vis/vis_bevcar.yaml'
```


## 👩‍⚖️  License

The code is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

We thank the authors of [Simple-BEV](https://github.com/aharley/simple_bev) for publicly releasing their [source code](https://github.com/aharley/simple_bev).

This work was funded by Qualcomm Technologies Inc., the German Research Foundation (DFG) Emmy Noether Program grant No 468878300, and an academic grant from NVIDIA.
<br><br>
<p float="left">
  <a href="https://www.qualcomm.com/"><img src="./assets/qualcomm_logo.png" alt="drawing" height="80"/></a>
  &nbsp;
  &nbsp;
  &nbsp;
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="DFG logo" height="100"/></a>
</p>
