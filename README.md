# Structurally Disentangled Feature Fields Distillation for 3D Understanding and Editing

## Installation
```
pip install torch torchvision
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```

## Example
In the data folder we have the configs files to segment the tabletop of the garden scene from MipNeRF-360, the weights, features and dataset should be download seperetly

Step to run our example:
* **Download** weights from [here](https://1drv.ms/u/s!AiZ2JMCsXZrobp7qn6y1WdIq3bY?e=UeB6j5).
* **Download** _MipNeRF-360_ Garden scene.
* **Download** indep-0.pt from [here](https://1drv.ms/u/s!AiZ2JMCsXZroa_kjhbOeo3XTXmk?e=M7L2AZ).
    * Make sure that indep-0.pt is in: `data\feat_imgs\garden`.
* Set environment variables.
* Run `python launch.py`.

### Set environment variables

You should set `CONFIG_PATH`, `RESUME` and `INPUT_DIR`
```
CONFIG_PATH=data\configs\tabletop_seg.yaml # or tabletop_rm_hl
RESUME=data\weights\garden.ckpt
INPUT_DIR=data\garden # download from MipNeRF-360 dataset
```

### Render
```
python launch.py \
  --config $CONFIG_PATH \ 
  --resume $RESUME \
  --gpu 0 \
  --exp_dir "./exp/" \
  --train \
  dataset.root_dir=$INPUT_DIR \
  trainer.max_steps=20000 \
  dataset.img_downscale=8 \
  dataset.traj_pos_mult=1 \
  system.train_features=True \
  model.only_features=True \
  dataset.load_features=False \
  dataset.n_test_traj_steps=90 \
  system.export_features=False \
  system.force_export_all=True \
  --trial_name="garden_seg"
```

### Results
#### Segmentation
[![Watch the video](https://raw.githubusercontent.com/StructurallyDisentangled/StructurallyDisentangled/main/data/renders/t0.png)](https://raw.githubusercontent.com/StructurallyDisentangled/StructurallyDisentangled/main/data/renders/garden-tabletop.mp4)
#### Highlight removal
[![Watch the video](https://raw.githubusercontent.com/StructurallyDisentangled/StructurallyDisentangled/main/data/renders/r0.png)](https://raw.githubusercontent.com/StructurallyDisentangled/StructurallyDisentangled/main/data/renders/garden-tabletop_rm_hl.mp4)

## Train new scene
### Train color
see `train.yaml` for config example

```
python launch.py --config /path/to/config.yaml --gpu 0 --train dataset.root_dir=/path/to/input_dir --trail_name=<optional> 
```

### Train feature
see `train_features.yaml` for config example
```
python launch.py --config /path/to/config.yaml --resume=/path/to/ckpt --resume_weights_only --gpu 0 --train dataset.root_dir=/path/to/input_dir --trail_name=<optional> 
```

## Segment and Edit
see `tabletop_seg.yaml` for config example for segmentation 
```
python launch.py --config /path/to/config.yaml --resume=/path/to/ckpt --gpu 0 --test dataset.root_dir=/path/to/input_dir --trail_name=<optional> 
``` 