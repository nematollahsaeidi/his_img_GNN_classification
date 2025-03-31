import os
import numpy as np
import torch
import random
import time
import timm
import tifffile
import torch.nn as nn
from huggingface_hub import login, hf_hub_download
from tqdm import tqdm
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

lr = 1e-4
num_epochs = 300
batch_size = 32
model_types = ['swin']  # Options: 'vit', 'uni', 'swin'
dataset_type = 'PanNuke'  # Options: 'PanNuke', 'BACH'

if dataset_type == 'PanNuke':
    benign_images = np.load(
        '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke_two_classes/images/benign_images.npy')
    benign_labels = np.load(
        '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke_two_classes/labels/benign_labels.npy')
    malignant_images = np.load(
        '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke_two_classes/images/malignant_images.npy')
    malignant_labels = np.load(
        '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke_two_classes/labels/malignant_labels.npy')

    all_images = np.concatenate([benign_images, malignant_images], axis=0)
    all_labels = np.concatenate([benign_labels, malignant_labels], axis=0)
elif dataset_type == 'BACH':
    base_path = '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/bach'
    categories = ['benign', 'insitu', 'invasive', 'normal']
    image_paths = []
    labels = []

    print(f"Loading {dataset_type} dataset...")
    for category in categories:
        folder_path = os.path.join(base_path, category)
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.tif'):
                image_paths.append(os.path.join(folder_path, img_file))
                labels.append(category)

    all_images = []
    for img_path in tqdm(image_paths):
        img = tifffile.imread(img_path)
        if img.dtype == np.uint16:  # Convert uint16 to uint8
            img = (img / 256).astype(np.uint8)
        if img.ndim == 2:  # If grayscale, convert to RGB
            img = np.stack([img] * 3, axis=-1)
        all_images.append(img)
    all_images = np.array(all_images)
    all_labels = labels
else:
    raise ValueError("Invalid dataset_type. Choose 'PanNuke' or 'BACH'.")

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)
num_classes = len(np.unique(all_labels_encoded))

train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    all_images, all_labels_encoded, test_size=0.2, random_state=seed, stratify=all_labels_encoded
)


def get_memory_size_mb(data):
    if isinstance(data, np.ndarray):
        return data.nbytes / (1024 * 1024)
    elif isinstance(data, torch.Tensor):
        return data.element_size() * data.nelement() / (1024 * 1024)
    return 0


def get_patch_embeddings(model, x, model_type='vit'):
    if model_type == 'vit':
        x = model._process_input(x)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        return x[:, 1:]
    elif model_type == 'uni':
        x = model.patch_embed(x)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + model.pos_embed
        x = model.blocks(x)
        x = model.norm(x)
        return x[:, 1:]
    elif model_type == 'swin':
        x = model.patch_embed(x)
        x = model.layers[0](x)
        x = model.layers[1](x)
        x = model.layers[2](x)
        return x
    else:
        raise ValueError("Unsupported model type. Choose 'vit', 'uni', or 'swin'.")


def extract_features(images, model, transform, model_type='vit'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    features = []
    with torch.no_grad():
        for img in images:
            img_tensor = transform(img).unsqueeze(0).to(device)
            patch_embeddings = get_patch_embeddings(model, img_tensor, model_type=model_type)
            patch_embeddings = patch_embeddings.squeeze(0).cpu().numpy()
            features.append(patch_embeddings)
    return features


def build_graph(patch_features, num_clusters=10, similarity_threshold=0.6):
    if len(patch_features.shape) > 2:
        patch_features = patch_features.reshape(-1, patch_features.shape[-1])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(patch_features)
    cluster_centers = kmeans.cluster_centers_
    sim_matrix = cosine_similarity(cluster_centers)
    edges = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if sim_matrix[i, j] > similarity_threshold:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(cluster_centers, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index), get_memory_size_mb(node_features) + get_memory_size_mb(
        edge_index)


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, model, transform, model_type='vit'):
        self.labels = torch.tensor(labels, dtype=torch.long)
        features = extract_features(images, model, transform, model_type=model_type)
        self.graphs = []
        graph_memory_sizes = []
        print("Starting build_graph...")
        for feat in tqdm(features):
            graph, memory_size = build_graph(feat)
            self.graphs.append(graph)
            graph_memory_sizes.append(memory_size)
        avg_graph_memory = np.mean(graph_memory_sizes)
        print(f"Average Graph Memory Usage of a processed image ({model_type}): {avg_graph_memory:.3f} MB")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


class GNN_GAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super(GNN_GAT, self).__init__()
        self.conv1 = GATConv(in_dim, 8, heads=heads, concat=True)
        self.conv2 = GATConv(8 * heads, 4, heads=heads, concat=True)
        self.fc = nn.Linear(4 * heads, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


class GNN_Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super(GNN_Transformer, self).__init__()
        self.conv1 = TransformerConv(in_dim, 8, heads=heads, concat=True)
        self.conv2 = TransformerConv(8 * heads, 4, heads=heads, concat=True)
        self.fc = nn.Linear(4 * heads, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


def train_gnn(model, train_loader, val_loader, num_epochs=300, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, labels).item()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}')
        print('-' * 25)

    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

for model_type in model_types:
    print(f"\nRunning with {model_type.upper()} model")
    model_name = f'graph_{model_type}'

    if model_type == 'vit':
        feature_model = models.vit_b_16(pretrained=True)
    elif model_type == 'uni':
        login('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')  # Replace with your HF token
        local_dir = "vit_large_patch16_224_uni_mass100k/"
        os.makedirs(local_dir, exist_ok=True)
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        feature_model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=False
        )
        feature_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
                                      strict=True)
    elif model_type == 'swin':
        feature_model = timm.create_model(
            'swin_large_patch4_window7_224',
            pretrained=True,
            num_classes=0
        )
    feature_model = feature_model.to(device)
    feature_model.eval()

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_results = {'acc': [], 'f1': [], 'balanced_acc': [], 'auc': []}
    train_times = []
    test_times = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images)):
        print(f'\nFold {fold + 1}')

        # Create datasets
        train_dataset = GraphDataset(
            train_val_images[train_idx], train_val_labels[train_idx],
            feature_model, train_transform, model_type=model_type
        )
        val_dataset = GraphDataset(
            train_val_images[val_idx], train_val_labels[val_idx],
            feature_model, val_test_transform, model_type=model_type
        )
        test_dataset = GraphDataset(
            test_images, test_labels,
            feature_model, val_test_transform, model_type=model_type
        )

        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        in_dim = 1024 if model_type == 'uni' else 768
        if model_type in ['vit', 'uni', 'swin']:
                model = GNN_GAT(in_dim=in_dim, out_dim=num_classes)
            # elif model_type == 'swin':
            # model = GNN_Transformer(in_dim=in_dim, out_dim=num_classes)

        start_time = time.time()
        model = train_gnn(model, train_loader, val_loader, num_epochs, lr)
        train_time = time.time() - start_time
        train_times.append(train_time)

        # Evaluate on test set
        start_time = time.time()
        test_preds, test_labels_list, test_probs = [], [], []
        model.eval()
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Shape: (batch_size, num_classes)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Shape: (batch_size,)
                test_preds.extend(preds)
                test_labels_list.extend(labels.numpy().flatten())
                test_probs.extend(probs)
        test_time = time.time() - start_time
        test_times.append(test_time)

        test_probs = np.array(test_probs)

        acc = accuracy_score(test_labels_list, test_preds)
        f1 = f1_score(test_labels_list, test_preds, average='macro')
        balanced_acc = balanced_accuracy_score(test_labels_list, test_preds)
        if num_classes == 2:
            auc = roc_auc_score(test_labels_list, test_probs[:, 1])  # Binary case
        else:
            auc = roc_auc_score(test_labels_list, test_probs, multi_class='ovr')  # Multi-class case

        test_results['acc'].append(acc)
        test_results['f1'].append(f1)
        test_results['balanced_acc'].append(balanced_acc)
        test_results['auc'].append(auc)

        print(
            f'Fold {fold + 1} | Test Acc: {acc:.4f}, Test F1: {f1:.4f}, Balanced Acc: {balanced_acc:.4f}, AUC: {auc:.4f}'
        )

    avg_train_time = np.mean(train_times)
    std_train_time = np.std(train_times)
    avg_test_time = np.mean(test_times)
    std_test_time = np.std(test_times)
    avg_test_acc = np.mean(test_results['acc'])
    std_test_acc = np.std(test_results['acc'])
    avg_test_f1 = np.mean(test_results['f1'])
    std_test_f1 = np.std(test_results['f1'])
    avg_test_balanced_acc = np.mean(test_results['balanced_acc'])
    std_test_balanced_acc = np.std(test_results['balanced_acc'])
    avg_test_auc = np.mean(test_results['auc'])
    std_test_auc = np.std(test_results['auc'])

    print(
        f'\nParameters: Seed={seed}, Learning Rate={lr}, Epochs={num_epochs}, Batch Size={batch_size}, Model={model_name}, Dataset={dataset_type}'
    )
    print(f'Average Test Accuracy: {avg_test_acc:.4f} (±{std_test_acc:.4f})')
    print(f'Average Test F1-Score: {avg_test_f1:.4f} (±{std_test_f1:.4f})')
    print(f'Average Test Balanced Accuracy: {avg_test_balanced_acc:.4f} (±{std_test_balanced_acc:.4f})')
    print(f'Average Test AUC: {avg_test_auc:.4f} (±{std_test_auc:.4f})')
    print(f'Average Train Time: {avg_train_time:.2f}s (±{std_train_time:.2f}s)')
    print(f'Average Test Time: {avg_test_time:.2f}s (±{std_test_time:.2f}s)')
