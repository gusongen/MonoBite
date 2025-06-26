#  MonoBite: Scale-Aware 3D Reconstruction and Volume Estimation from Monocular Multi-Food Images

This project is the 1st place solution for [CVPR 2025 - 2nd MetaFood Workshop](https://sites.google.com/view/cvpr-metafood-2025/challenge-1) Challenge 1 (3D Reconstruction From Monocular Multi-Food Images). The core script `main.py` can automatically process multiple food instances in batch, reconstruct multiple food items from a single RGB image, and output the 3D volume prediction for each food, supporting food nutrition analysis and health management.

## Dependencies
Please ensure the following Python libraries are installed:
- numpy
- pandas
- trimesh
- open3d
- opencv-python
- matplotlib
- tqdm

Install with:
```bash
pip install numpy pandas trimesh open3d opencv-python matplotlib tqdm
```

## Output Files
- `submit.csv`: Final submission result, containing the predicted volume for each food
- `submit_model/`: Directory saving all processed mesh files

## Quick Start
1. Run the main script:
   ```bash
   python main.py
   ```
2. After execution, `submit.csv` and the `submit_model/` folder will be generated in the current directory.

## Directory Structure Example
```
├── main.py
├── meta_new.csv
├── submit.csv
├── submit_model/
│   ├── 1.obj
│   ├── 2.obj
│   └── ...
└── ...
```

## Result Example
The `submit.csv` file is formatted as follows:
```
id,predicted
1,123.45
2,67.89
...
```
