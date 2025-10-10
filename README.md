
# Integrating Foundation Model Features into Graph Neural Network and Fusing Predictions with Standard Fine-Tuned Models for Histology Image Classification

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
- **Dataset Generation**:
To generate the PanNuke dataset, run:
```sh
python pannuke_binary_data_generation.py
```

### 2. BACH Dataset
- **Description**: The Breast Cancer Histology (BACH) dataset, containing four categories: `benign`, `insitu`, `invasive`, and `normal`.
- **Files**: Images are stored as `.tif` files in category-specific folders under `/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/bach`.


### 3. BreakHis Dataset
- **Description**: The Breast Histological Images for Classification (BreakHis) dataset, a binary classification dataset categorized into benign and malignant. Images are available at multiple magnification levels: 40X, 100X, 200X, 400X, and 4M(combination of 4 magnifications).
- **Files**: 
  - **Benign Images**: `benign_images.npy`
  - **Benign Labels**: `benign_labels.npy`
  - **Malignant Images**: `malignant_images.npy`
  - **Malignant Labels**: `malignant_labels.npy`
  - **Path**: Files are stored under /mnt/miaai/STUDIES/his_img_GNN_classification/datasets/breakhis_two_classes/{magnification}/, where {magnification} is one of 40X, 100X, 200X, 400X, or 4M.
 

## Model Architectures

### 1. Base Models (`base_models_classification.py`)
The base models approach supports the following pre-trained architectures:
- **ViT-B/16** (Vision Transformer)
- **VGG19**
- **EfficientNet-V2-S**
- **DenseNet201**

Each model is initialized with pre-trained weights, and the final classification layer is modified to match the number of classes in the dataset (2 for PanNuke, 4 for BACH, 2 for BreakHis).

### 2. Graph-Based Models (`graph_models_classification.py`)
The graph-based approach combines pre-trained feature extractors with GNNs:
- **Feature Extractors**:
  - **ViT-B/16** (Vision Transformer)
  - **UNI** (MahmoodLab/UNI)
  - **UNI2-h** (MahmoodLab/UNI2-h)
  <!--- **Swin Transformer** (swin_large_patch4_window7_224)-->
  - **CONCH** (conch_ViT-B-16)
  - **DenseNet201**
  - **VGG19**
  - **EfficientNet-V2-S**
- **Graph Construction**:
  - Patch embeddings are extracted from the feature extractor.
  - K-Means clustering (optional, with `num_clusters=10` or `None`) groups patches into nodes.
  - Cosine similarity (threshold=0.6) determines edges between nodes.
- **GNN Models**:
  - **GAT (Graph Attention Network)**: Uses attention mechanisms to weigh node relationships.
  - **TransformerConv**: Employs transformer-based convolution for graph processing.

## Performance Metrics
Both approaches report the following metrics:
<!--- **Accuracy**: Overall classification accuracy.-->
- **F1-Score (Macro Average)**: Balances precision and recall across classes.
- **Balanced Accuracy**: Accounts for class imbalance.
<!--- **AUC (Area Under ROC Curve)**: Measures discriminative ability (binary or multi-class OVR).-->
- **Training Time**: Time taken to train the model.
- **Testing Time**: Time taken to evaluate on the test set.
<!--- **Confusion Matrix**: Summarizes classification performance across folds.-->

## Running the Scripts

### Base Models
To train and evaluate the base models:
```sh
python base_models_classification.py
```
- Edit `models_to_train` to select a model (e.g., `'densenet201'`, `'vgg19'`, `'efficientnet_v2_s'`, `'vit'`).
- Set `dataset_type` to `'PanNuke'` or `'BACH'` or `'BreakHis'`.

### Graph-Based Models
To train and evaluate the graph-based models:
```sh
python graph_models_classification.py
```
- Edit `model_type` to select a feature extractor (e.g., `'uni2'`, `'vit'`, `'swin'`, `'conch'`, `'densenet201'`).
- Set `dataset_type` to `'PanNuke'` or `'BACH'` or `'BreakHis'`.
- Set `num_clusters` to `10` or `None` for graph construction.
- For UNI or UNI2-h, provide a Hugging Face token via `login()`.

## Additional Notes
- **Pre-trained Weights**: Paths to pre-trained weights are specified in the scripts (e.g., `/mnt/miaai/STUDIES/his_img_GNN_classification/pretrain_model_weights/`).
- **Logging**: Training and evaluation logs are saved to `/mnt/miaai/Nemat/nuclei_seg/best_models/` for graph-based models.
- **Model Saving**: Best models (based on validation loss) are saved per fold in `best_models/{model_name}_{dataset_type}/`.

--- 

### Contact
Emad (Nematollah) Saeidi

Email: nemat.saeidi@gmail.com, Nematollah.Saeidi@dp-uni.ac.at
