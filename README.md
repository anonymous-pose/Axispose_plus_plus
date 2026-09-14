# AxisPose++

## Setup

```bash
conda create -n axisposepp python=3.12 -y
conda activate axisposepp
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0 torchvision==0.18.0
pip install -r requirements.txt
```

## Test

```bash
python scripts/test_shapenet.py --config configs/test_shapenet.yaml
```

The command evaluates the included 100 randomly sampled ShapeNet pairs. Results and bounding-box visualizations are saved in `outputs/test_shapenet/`.

## Data layout

The full ShapeNet data is available from the [NOPE repository](https://github.com/nv-nguyen/nope).

```text
data/shapenet/
├── train_pairs.pkl
├── test_pairs.pkl
├── bbox_3d.json
├── train/
│   └── <object_id>/
│       ├── <view_id>_rot.png
│       ├── <view_id>_axisRot.png
│       └── <view_id>_pose.txt
└── test/
    └── <object_id>/
        ├── <view_id>_rot.png
        ├── <view_id>_axisRot.png
        └── <view_id>_pose.txt
```

Each pair file is a Python pickle containing dictionaries with `ref` and `query` prefixes. The pose and projected tri-axis files must correspond to the RGB image.

## Train

Prepare the full ShapeNet split in the layout above, set `data.root` in `configs/train_shapenet.yaml`, then run:

```bash
python scripts/train.py --config configs/train_shapenet.yaml
```
