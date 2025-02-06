# Structurally Disentangled Feature Fields Distillation for 3D Understanding and Editing

## Installation
```
pip install torch torchvision
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```

## How to Train
Train scene on color
see configs/sem_unisdf.yaml for config example

```
python launch.py --config /path/to/config.yaml --gpu 0 --train dataset.root_dir=/path/to/input_dir --trail_name=<optional> 
```

Train scene on feature
see configs/sem_unisdf_features.yaml for config example
```
python launch.py --config /path/to/config.yaml --resume=/path/to/ckpt --resume_weights_only --gpu 0 --train dataset.root_dir=/path/to/input_dir --trail_name=<optional> 
```

## Segment and Edit
see configs/seg_example.yaml for config example for segmentation 
```
python launch.py --config /path/to/config.yaml --resume=/path/to/ckpt --gpu 0 --test dataset.root_dir=/path/to/input_dir --trail_name=<optional> 
``` 