# AxisPose++

## Setup

```bash
conda create -n axisposepp python=3.12 -y
conda activate axisposepp
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.0 torchvision==0.18.0
pip install -r requirements.txt
```

## Test

Download [axisposepp_shapenet_weights.zip](https://github.com/anonymous-pose/Axispose_plus_plus/releases/download/addfile/axisposepp_shapenet_weights.zip), place it in the repository root, and extract it:

```bash
unzip axisposepp_shapenet_weights.zip
```

After extraction, the repository contains:

```text
AxisPose++/
├── checkpoints/
│   ├── axisposepp_shapenet.pt
│   └── dinov2_vits14_reg4_pretrain.pth
├── configs/
├── data/
├── axisposepp/
└── scripts/
```

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
## Large-rotation evaluation

Use the same environment and pretrained checkpoints described above. Download [shapenet_large_rotation_test.zip](https://github.com/anonymous-pose/Axispose_plus_plus/releases/download/addfile_large_rotation/shapenet_large_rotation_test.zip) and place it in the repository root. The archive contains the test script, configuration, and 200 ShapeNet reference-target pairs: 50 in each of (45, 60], (60, 90], (90, 120], and (120, 150] degrees.

From the repository root, extract the archive and merge its files:

```bash
unzip shapenet_large_rotation_test.zip
cp -r shapenet_large_rotation_test/configs shapenet_large_rotation_test/scripts shapenet_large_rotation_test/data .
```

The added files should have the following layout:

```text
configs/test_shapenet_large_rotation.yaml
scripts/test_shapenet_large_rotation.py
data/shapenet_large_rotation/
├── test_pairs.pkl
├── bbox_3d.json
└── test/
    └── <object_id>/
        ├── <view_id>_rot.png
        ├── <view_id>_axisRot.png
        └── <view_id>_pose.txt
```

Run the evaluation from the repository root:

```bash
python scripts/test_shapenet_large_rotation.py --config configs/test_shapenet_large_rotation.yaml
```

The configuration uses the same pretrained model without retraining and disables the reference-relative rotation constraint in candidate selection. Per-interval ACC@30 and median angular error are printed at completion. Results and bounding-box visualizations are saved in `outputs/test_shapenet_large_rotation/`, with per-interval metrics in `summary_by_angle.csv` and `summary_by_angle.json`.
