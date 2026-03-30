# PoseCastFormer

PoseCastFormer is a state-of-the-art 3D human pose estimation model that leverages transformer architecture for accurate 3D pose estimation from 2D keypoints.

##  Table of Contents

- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Wild Inference](#wild-inference)
- [Visualization](#visualization)
- [License](#license)

##  Dependencies

The project requires the following dependencies:

```bash
# Core dependencies
numpy
torch
tensorboardX
pyyaml
argparse
pickle
matplotlib
opencv-python
scipy
scikit-learn
prettytable
tqdm

# Additional dependencies for visualization
ffmpeg-python  # for video rendering
```

Install the required packages using pip:

```bash
pip install numpy torch tensorboardX pyyaml matplotlib opencv-python scipy scikit-learn prettytable tqdm ffmpeg-python
```

##  Dataset

The model requires the following data files:

### Required Data Files

1. **3D Human3.6M Dataset**:
   - `data/data_3d_h36m.npz` - 3D pose annotations
   - `data/data_2d_h36m_gt.npz` - Ground truth 2D poses
   - `data/data_2d_h36m_cpn_ft_h36m_dbb.npz` - CPN detected 2D poses
   - `data/motion3d/h36m_sh_conf_cam_source_final.pkl` - Preprocessed motion data

For the three files data_3d_h36m.npz, data_2d_h36m_gt.npz, and data_2d_h36m_cpn_ft_h36m_dbb.npz, please refer to the official dataset setup guide from VideoPose3D:  
[https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/main/DATASETS.md)

For the file h36m_sh_conf_cam_source_final.pkl, please refer to the data preparation guide from MotionBERT:  
[https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)

### Data Directory Structure

```
data/
├── data_3d_h36m.npz
├── data_2d_h36m_gt.npz
├── data_2d_h36m_cpn_ft_h36m_dbb.npz
└── motion3d/
   └── h36m_sh_conf_cam_source_final.pkl
   ├── MB3D_f243s81/
      └──H36M-CPN
      ├──H36M-SH
      
```

### Training Commands

**Basic Training:**
```bash
 python train.py --config configs/pose3d/MB_train_h36m_posecastformer_scratch_SH.yaml --checkpoint checkpoint/pose3d/posecastformer_scratch_SH                                                
                
```
### Training Configuration Options

- `--config`: Path to configuration file
- `--checkpoint`: Checkpoint directory for saving models
dom seed for reproducibility

##  Evaluation

### Using test_metrics.py

Evaluate model performance on Human3.6M dataset:

```bash
python test_metrics.py --config configs/pose3d/MB_train_h36m_posecastformer_scratch_SH.yaml --checkpoint checkpoint/pose3d/posecastformer_scratch_SH/best_epoch.bin --log result\test_results_H36M.log 
```

**Parameters:**
- `--config`: Configuration file path
- `--checkpoint`: Model checkpoint to evaluate (required)

**Output Metrics:**
- **MPJPE** (Mean Per-Joint Position Error)
- **MPJVE** (Mean Per-Joint Velocity Error)
- **PA-MPJPE** (Procrustes-Aligned MPJPE)

##  Wild Inference

### Using test_3dpw.py

Perform inference on in-the-wild 3DPW dataset:

```bash
python test_3dpw.py --config configs/pose3d/MB_train_h36m_posecastformer_scratch_SH.yaml --checkpoint checkpoint/posecastformer_scratch_SH/best_epoch.bin --log  result\results_3dpw.log             
```

**Parameters:**
- `--config`: Configuration file path
- `--checkpoint`: Model checkpoint for inference
- `--log`: Results log file


**Note:** Update the 3DPW dataset path in `test_3dpw.py`:
```python
data_root = "F:/3DPW/sequenceFiles/sequenceFiles/test"  # Change to your 3DPW path
```

##  Visualization

### Using visualization/run.py

Generate visualizations and animations:

```bash
# Basic visualization(example)
 python visualization/run.py -k cpn_ft_h36m_dbb -c checkpoint --evaluate best_epoch.bin --backbone posecastformer --render --viz-subject S9 --viz-action "SittingDown" --viz-camera 0 --viz-video "F:H36M\S9\Videos\SittingDown.55011271.mp4" --viz-output output.mp4
```

For detailed explanations of all visualization parameters, please refer to the VideoPose3D documentation:  
[https://github.com/facebookresearch/VideoPose3D/blob/main/DOCUMENTATION.md](https://github.com/facebookresearch/VideoPose3D/blob/main/DOCUMENTATION.md)


##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 PoseCastFormer Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


