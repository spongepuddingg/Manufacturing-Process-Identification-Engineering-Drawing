# Graph neural network-enabled manufacturing method classification from engineering drawings

This repository contains two complementary deep learning models for manufacturing process analysis using Graph Neural Networks (GNNs). The project consists of two main components:

1. **Centerline Detection and Annotation** (`centerline_ML/`)
2. **Manufacturing Method Identification** (`manufacturing_method_identification/`)

## Project Overview

The system processes manufacturing drawings (SVG format) to:
- Extract and classify centerlines from technical drawings
- Identify manufacturing methods (Lathe, Sheet Metal, Milling) from geometric features

## Repository Structure

```
manufacturing_process_gnn/
├── centerline_ML-main/                    # Centerline detection module
│   ├── gcn_lib/                          # Graph convolution library
│   │   ├── dense/                        # Dense graph operations
│   │   └── sparse/                       # Sparse graph operations
│   ├── processed/                        # Output directory for processed data
│   ├── saved_model/                      # Trained model weights
│   ├── main.py                          # Training script
│   ├── model.py                         # DeepGCN model definition
│   ├── inference.py                     # Inference script
│   ├── svg_2_npy.py                     # SVG to numpy conversion
│   └── learning_helper.py               # Training utilities
└── manufacturing_method_identification-main/  # Manufacturing method classification
    ├── gcn_lib/                         # Graph convolution library
    ├── saved_model/                     # Trained model weights
    ├── main.py                          # Training script
    ├── model.py                         # Base GNN model
    ├── classifier.py                    # Classification model
    ├── inference.py                     # Inference script
    ├── clusterData.py                   # Data loading and preprocessing
    ├── data_loader.py                   # Alternative data loader
    ├── learning_helper.py               # Training utilities
    └── len_group.py                     # Length grouping utilities
```

## Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU 

### Dependencies
Install the required packages using conda:

```bash
# Install PyTorch (version 1.9.0)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install PyTorch Geometric
conda install pytorch-geometric -c rusty1s -c conda-forge

# Additional dependencies
pip install svgpathtools
pip install scikit-learn
pip install pandas
pip install tqdm
pip install torch-cluster
pip install torch-scatter
```

## Data Format

### Input Data
- **SVG Files**: Technical drawings in SVG format
- **Expected Structure**: Line segments with coordinates and optional styling information

### Data Processing
1. **SVG to NumPy**: Use `svg_2_npy.py` to convert SVG files to numpy arrays
2. **Data Organization**: 
   - Lathe parts: `data/Lathe/*.npy`
   - Sheet Metal parts: `data/SheetMetal/*.npy`
   - Milling parts: `data/Milling/*.npy`
3. **Labels**: Manufacturing method labels stored in `data/labels.pkl`

## Usage

### 1. Centerline Detection and Annotation

#### Training
```bash
cd centerline_ML-main/
python main.py --batch_size 16 --epochs 200 --lr 1e-3 --n_classes 3
```

#### Key Parameters:
- `--batch_size`: Mini-batch size (default: 16)
- `--in_channels`: Input feature dimensions (default: 4)
- `--n_classes`: Number of centerline classes (default: 3)
- `--k`: Number of nearest neighbors for graph construction (default: 16)
- `--n_blocks`: Number of GCN blocks (default: 6)
- `--block`: Block type {plain, res, dense} (default: plain)

#### Inference
```bash
python inference.py --pretrained_model saved_model/best_test.pt --batch_size 16
```

### 2. Manufacturing Method Identification

#### Training
```bash
cd manufacturing_method_identification-main/
python main.py --batch_size 24 --lr 5e-4 --n_classes 3
```

#### Key Parameters:
- `--batch_size`: Mini-batch size (default: 24)
- `--n_classes`: Number of manufacturing methods (default: 3)
- `--in_channels`: Input feature dimensions (default: 5)
- `--hidden_channels`: Hidden layer size (default: 256)
- `--feat_dims`: Feature dimensions (default: 128)

#### Inference
```bash
python inference.py --pretrained_model saved_model/best_test.pt --batch_size 24
```

## Model Architecture

### Centerline Detection
- **Input**: Representation of line segments (x1, y1, x2, y2)
- **Architecture**: Deep Graph Convolutional Network with configurable blocks
- **Output**: Centerline classification for each line segment
- **Graph Construction**: k-NN graph with dilated connections

### Manufacturing Method Classification
- **Input**: Graph representation of manufacturing drawings
- **Architecture**: 
  - LSTM for sequential feature extraction
  - Dense SAGE convolution layers
  - Differentiable pooling for graph-level classification
- **Output**: Manufacturing method classification (Lathe/Sheet Metal/Milling)

## Data Preprocessing

### SVG Processing
```python
# Convert SVG files to numpy format
python svg_2_npy.py
```

### Graph Construction
- **Centerline**: k-NN graph based on spatial proximity
- **Manufacturing**: Custom graph based on stroke relationships and geometric features

## Training Details

### Centerline Detection
- **Loss**: Cross-entropy loss
- **Optimizer**: Adam (lr=1e-3)
- **Data Split**: 80% train, 20% test
- **Augmentation**: Random translation, normalization

### Manufacturing Classification
- **Loss**: Cross-entropy + link loss + entropy loss
- **Optimizer**: Adam (lr=5e-4)
- **Data Split**: 80% train, 20% test
- **Features**: Geometric and topological features

## Output

### Centerline Detection
- Processed files saved in `processed/` directory
- Format: Original coordinates + predicted centerline labels

### Manufacturing Classification
- Class predictions for input drawings
- Confidence scores for each manufacturing method

## Model Performance

Models are saved when achieving best balanced accuracy on test set. Training logs show:
- Training accuracy and loss
- Test accuracy and balanced accuracy
- Model checkpoints saved automatically

## Troubleshooting

### Common Issues
1. **CUDA Memory**: Reduce batch size if encountering OOM errors
2. **Data Loading**: Ensure data directory structure matches expected format
3. **Dependencies**: Verify PyTorch Geometric installation with correct CUDA version

### Data Requirements
- Minimum 500 samples per class for training from scratch
- SVG files should contain line segments with coordinate information
- Labels file must match filename patterns in data directories

## Citation

If you use this code in your research, please cite the relevant papers and acknowledge the original implementation.

@article{XIE2022103697,
title = {Graph neural network-enabled manufacturing method classification from engineering drawings},
journal = {Computers in Industry},
volume = {142},
pages = {103697},
year = {2022},
issn = {0166-3615},
doi = {https://doi.org/10.1016/j.compind.2022.103697},
url = {https://www.sciencedirect.com/science/article/pii/S016636152200094X},
author = {Liuyue Xie and Yao Lu and Tomotake Furuhata and Soji Yamakawa and Wentai Zhang and Amit Regmi and Levent Kara and Kenji Shimada},
keywords = {Engineering drawing, Manufacturing method, Image classification, Vectorization, Graph neural network, Hierarchical graph learning}
}

## License

See individual LICENSE files in each subdirectory for specific licensing information.