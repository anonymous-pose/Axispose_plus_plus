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

## Reference Coordinate Frame Sensitivity

This experiment measures how AxisPose++ responds to a change in the object coordinate frame at test time. It uses the same 100 public ShapeNet reference–target pairs in all conditions.

Complete the repository setup and extract the released ShapeNet checkpoints as described above. Then run the following commands from the repository root:

```bash
python -m pip install numpy==1.26.4 opencv-python-headless==4.11.0.86 "Pillow>=10.1"
python scripts/test_reference_frame.py
```

For each pair, we sample a 3D rotation axis with a fixed random seed (`2027`) and use the same axis at 0°, 15°, 30°, and 60°. Let `Q` be the rotation defining the new object coordinate frame. We update the reference rotation to `R_ref @ Q.T` and reproject the reference tri-axis. Both RGB images, camera intrinsics, and translations remain fixed. The target annotation is expressed in the new frame as `R_target @ Q.T`, which preserves the ground-truth relative rotation. We rerun inference for each condition and multiply the predicted target rotation on the right by `Q` before measuring its angular error in the original frame.

Results are saved to `outputs/reference_frame/`:

- `summary.csv` contains aggregate metrics and changes relative to 0°.
- `paired_errors.csv` contains per-pair angular errors and prediction differences relative to the 0° condition.
- `q0/`, `q15/`, `q30/`, and `q60/` contain the selected poses and per-pair frame transformations.
- `figures/` contains four illustrative comparisons and a combined `overview.png`. The target overlays use ground-truth translation for both boxes to visualize rotation differences.
- `protocol.json` records the run settings, software versions, and test pair list checksum.

Use `--all-figures` to save a panel for every test pair. For a quick check on four pairs, run:

```bash
python scripts/test_reference_frame.py --max-samples 4 --angles 0 60 --out outputs/reference_frame_smoke
```
