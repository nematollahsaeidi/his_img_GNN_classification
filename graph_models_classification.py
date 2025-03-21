import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import random
import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

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


def get_patch_embeddings(model, x):
    x = model._process_input(x)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    x = model.encoder(x)
    return x[:, 1:]


def extract_features(images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vit_b_16(pretrained=True)
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    features = []
    with torch.no_grad():
        for img in images:
            img_tensor = transform(img).unsqueeze(0).to(device)
            patch_embeddings = get_patch_embeddings(model, img_tensor)
            patch_embeddings = patch_embeddings.squeeze(0).cpu().numpy()
            features.append(patch_embeddings)
    return features


def build_graph(patch_features, num_clusters=10, similarity_threshold=0.6):
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
    def __init__(self, images, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)
        features = extract_features(images)
        self.graphs = []
        graph_memory_sizes = []
        for feat in features:
            graph, memory_size = build_graph(feat)
            self.graphs.append(graph)
            graph_memory_sizes.append(memory_size)
        avg_graph_memory = np.mean(graph_memory_sizes)
        print(f"Average Graph Memory Usage: {avg_graph_memory:.3f} MB")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super(GNN, self).__init__()
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

def train_gnn(model, train_loader, val_loader, num_epochs=300, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
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

        print(
            f'Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | \nVal Loss: {val_loss / len(val_loader):.4f}')

    train_time = time.time() - start_time
    return model, train_time

lr = 1e-4
num_epochs = 300
batch_size = 32
model_name = 'vit'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
test_results = {'acc': [], 'f1': [], 'balanced_acc': [], 'auc': []}
train_times = []
test_times = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images)):
    print(f'\nFold {fold + 1}')

    train_dataset = GraphDataset(train_val_images[train_idx], train_val_labels[train_idx])
    val_dataset = GraphDataset(train_val_images[val_idx], train_val_labels[val_idx])
    test_dataset = GraphDataset(test_images, test_labels)

    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GNN(in_dim=768, out_dim=num_classes)
    model, train_time = train_gnn(model, train_loader, val_loader, num_epochs, lr)
    train_times.append(train_time)

    test_preds, test_labels_list, test_probs = [], [], []
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(labels.numpy())
            test_probs.extend(probs)
    test_time = time.time() - start_time
    test_times.append(test_time)

    acc = accuracy_score(test_labels_list, test_preds)
    f1 = f1_score(test_labels_list, test_preds, average='macro')
    balanced_acc = balanced_accuracy_score(test_labels_list, test_preds)
    auc = roc_auc_score(test_labels_list, test_probs)

    test_results['acc'].append(acc)
    test_results['f1'].append(f1)
    test_results['balanced_acc'].append(balanced_acc)
    test_results['auc'].append(auc)

    print(f'Fold {fold + 1} | Test Acc: {acc:.4f}, Test F1: {f1:.4f}, Balanced Acc: {balanced_acc:.4f}, AUC: {auc:.4f}')

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

print(f'\nParameters: Seed={seed}, Learning Rate={lr}, Epochs={num_epochs}, Batch Size={batch_size}, Model={model_name}')
print(f'Average Train Time: {avg_train_time:.2f}s (±{std_train_time:.2f}s)')
print(f'Average Test Time: {avg_test_time:.2f}s (±{std_test_time:.2f}s)')
print(f'Average Test Accuracy: {avg_test_acc:.4f} (±{std_test_acc:.4f})')
print(f'Average Test F1-Score: {avg_test_f1:.4f} (±{std_test_f1:.4f})')
print(f'Average Test Balanced Accuracy: {avg_test_balanced_acc:.4f} (±{std_test_balanced_acc:.4f})')
print(f'Average Test AUC: {avg_test_auc:.4f} (±{std_test_auc:.4f})')