
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
import sys

# Set seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load data (unchanged)
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


# Dataset and transforms (unchanged)
class MedicalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        if self.transform:
            image = self.transform(torch.tensor(image))
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


# Model initialization (unchanged)
def initialize_model(model_name, num_classes):
    model = None
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=True)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    return model


# Training function with time tracking
def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_model_state = None
    best_f1 = 0.0
    train_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_list.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels_list, train_preds)
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step(val_f1)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}   | Val F1: {val_f1:.4f}')
        print('-' * 60)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    return best_model_state, train_time


# Main training and evaluation loop with added metrics and parameters
models_to_train = ['vit', 'densenet201', 'vgg19', 'efficientnet']
k_folds = 5
batch_size = 32
num_epochs = 30
lr = 1e-4
train_transform, val_transform = get_transforms()

test_results = {model_name: {'acc': [], 'f1': [], 'balanced_acc': [], 'auc': [], 'train_time': [], 'test_time': []}
                for model_name in models_to_train}

kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images)):
    print(f'\nFold {fold + 1}/{k_folds}')
    print('=' * 50)

    train_images = train_val_images[train_idx]
    train_labels = train_val_labels[train_idx]
    val_images = train_val_images[val_idx]
    val_labels = train_val_labels[val_idx]

    train_dataset = MedicalDataset(train_images, train_labels, train_transform)
    val_dataset = MedicalDataset(val_images, val_labels, val_transform)
    test_dataset = MedicalDataset(test_images, test_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for model_name in models_to_train:
        print(f'\nTraining {model_name.upper()}...')
        print(
            f'Parameters: Seed={seed}, Learning Rate={lr}, Epochs={num_epochs}, Batch Size={batch_size}, Model={model_name}')
        model = initialize_model(model_name, num_classes)
        best_model_state, train_time = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)

        model.load_state_dict(best_model_state)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        test_start_time = time.time()
        test_preds, test_labels_list, test_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, 1).cpu().numpy()
                test_preds.extend(preds)
                test_labels_list.extend(labels.numpy())
                test_probs.extend(probs)

        test_end_time = time.time()
        test_time = test_end_time - test_start_time

        test_acc = accuracy_score(test_labels_list, test_preds)
        test_f1 = f1_score(test_labels_list, test_preds, average='macro')
        test_balanced_acc = balanced_accuracy_score(test_labels_list, test_preds)
        test_auc = roc_auc_score(test_labels_list, test_probs)

        test_results[model_name]['acc'].append(test_acc)
        test_results[model_name]['f1'].append(test_f1)
        test_results[model_name]['balanced_acc'].append(test_balanced_acc)
        test_results[model_name]['auc'].append(test_auc)
        test_results[model_name]['train_time'].append(train_time)
        test_results[model_name]['test_time'].append(test_time)

        print(f'{model_name.upper()} - Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, '
              f'Test Balanced Acc: {test_balanced_acc:.4f}, Test AUC: {test_auc:.4f}')
        print(f'Train Time: {train_time:.2f}s, Test Time: {test_time:.2f}s')

# Final summary with all metrics
print('\nFinal Summary Across 5 Folds:')
print('=' * 50)
for model_name in models_to_train:
    sample_image, _ = train_dataset[0]  # a sample
    image_memory_size = sys.getsizeof(sample_image.storage()) / (1024 * 1024)
    avg_test_acc = np.mean(test_results[model_name]['acc'])
    std_test_acc = np.std(test_results[model_name]['acc'])
    avg_test_f1 = np.mean(test_results[model_name]['f1'])
    std_test_f1 = np.std(test_results[model_name]['f1'])
    avg_test_balanced_acc = np.mean(test_results[model_name]['balanced_acc'])
    std_test_balanced_acc = np.std(test_results[model_name]['balanced_acc'])
    avg_test_auc = np.mean(test_results[model_name]['auc'])
    std_test_auc = np.std(test_results[model_name]['auc'])
    avg_train_time = np.mean(test_results[model_name]['train_time'])
    avg_test_time = np.mean(test_results[model_name]['test_time'])

    print(f'{model_name.upper()}:')
    print(
        f'Parameters: Seed={seed}, Learning Rate={lr}, Epochs={num_epochs}, Batch Size={batch_size}, Model={model_name}')
    print(f'Average Test Accuracy: {avg_test_acc:.4f} (±{std_test_acc:.4f})')
    print(f'Average Test F1: {avg_test_f1:.4f} (±{std_test_f1:.4f})')
    print(f'Average Test Balanced Accuracy: {avg_test_balanced_acc:.4f} (±{std_test_balanced_acc:.4f})')
    print(f'Average Test AUC: {avg_test_auc:.4f} (±{std_test_auc:.4f})')
    print(f'Average Train Time: {avg_train_time:.2f}s')
    print(f'Average Test Time: {avg_test_time:.2f}s')
    print(f'Memory size of a processed image (training/testing): {image_memory_size:.2f} MB')
    print('-' * 50)
