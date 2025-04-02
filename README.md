
# Leveraging Medical Foundation Model Features in Graph Neural Network-Based Retrieval of Breast Histopathology Images

## Description
This repository contains a deep learning pipeline for classifying histopathology images. Two main approaches are implemented:

1. **Base Models Approach (`base_models_classification.py`)**: Utilizes pre-trained convolutional neural networks (CNNs) and vision transformers for direct image classification.
2. **Graph-Based Approach (`graph_models_classification.py`)**: Employs graph neural networks (GNNs) combined with pre-trained feature extractors to model relationships between image patches.

Both implementations leverage PyTorch, Torchvision, and additional libraries (e.g., PyTorch Geometric for GNNs) for training and evaluation. The pipelines support K-Fold cross-validation and provide detailed performance metrics.

## Datasets
The project supports two histopathology datasets:

### 1. PanNuke Dataset
- **Description**: A binary classification dataset derived from the PanNuke dataset, categorized into `benign` and `malignant` based on the proportion of neoplastic cells.
- **Files**:
  - **Benign Images**: `benign_images.npy`
  - **Benign Labels**: `benign_labels.npy`
  - **Malignant Images**: `malignant_images.npy`
  - **Malignant Labels**: `malignant_labels.npy`

### 2. BACH Dataset
- **Description**: The Breast Cancer Histology (BACH) dataset, containing four categories: `benign`, `insitu`, `invasive`, and `normal`.
- **Files**: Images are stored as `.tif` files in category-specific folders under `/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/bach`.

## Dataset Generation
To generate the PanNuke dataset, run:
```sh
python pannuke_binary_data_generation.py
```

## Model Architectures

### 1. Base Models (`base_models_classification.py`)
The base models approach supports the following pre-trained architectures:
- **VGG19**
- **EfficientNet-V2-S**
- **DenseNet201**

Each model is initialized with pre-trained weights, and the final classification layer is modified to match the number of classes in the dataset.

### 2. Graph-Based Models (`graph_models_classification.py`)
The graph-based approach combines pre-trained feature extractors with GNNs:
- **Feature Extractors**:
  - **ViT-B/16**
  - **UNI**
  - **Swin Transformer**
- **Graph Construction**: 
  - Patch embeddings are extracted from the feature extractor.
  - K-Means clustering groups patches into nodes, and cosine similarity determines edges.
- **GNN Models**:
  - **GAT (Graph Attention Network)**: Uses attention mechanisms to weigh node relationships.

## Training Strategy
- **Dataset Split**: Both scripts split the data into **train/validation (80%)** and **test (20%)**, with stratification based on labels.
- **Cross-Validation**: 5-Fold Cross-Validation is used to evaluate model performance on the train/validation set.

## Performance Metrics
Both approaches report the following metrics:
- **Accuracy**: Overall classification accuracy.
- **F1-Score (Macro Average)**: Balances precision and recall across classes.
- **Balanced Accuracy**: Accounts for class imbalance.
- **AUC (Area Under ROC Curve)**: Measures discriminative ability (binary or multi-class OVR).
- **Training Time**: Time taken to train the model.
- **Testing Time**: Time taken to evaluate on the test set.
- **Memory Usage**: Memory of processed images or graphs.

## Running the Scripts

### Base Models
To train and evaluate the base models:
```sh
python base_models_classification.py
```
- Edit `models_to_train` to select models (e.g., `['vgg19', 'efficientnet']`).
- Set `dataset_type` to `'PanNuke'` or `'BACH'`.

### Graph-Based Models
To train and evaluate the graph-based models:
```sh
python graph_models_classification.py
```
- Edit `model_types` to select feature extractors (e.g., `['swin', 'vit']`).
- Set `dataset_type` to `'PanNuke'` or `'BACH'`.
- For UNI, provide a Hugging Face token via `login()`.

--- 
