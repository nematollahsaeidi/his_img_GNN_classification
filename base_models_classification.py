from tqdm import tqdm
import time
import sys
import os
import tifffile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

k_folds = 5
batch_size = 32
num_epochs = 300
lr = 1e-4
models_to_train = 'densenet201'  # Options: 'vit', 'densenet201', 'vgg19', 'efficientnet_v2_s'
dataset_type = 'BACH'  # Options: 'PanNuke', 'BACH'

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


def initialize_model(model_name, num_classes):
    model = None
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=False)
        model.load_state_dict(torch.load("/mnt/miaai/STUDIES/his_img_GNN_classification/pretrain_model_weights/vgg19-dcbb9e9d.pth"))
        in_features = model.classifier[0].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(pretrained=False)
        model.load_state_dict(torch.load("/mnt/miaai/STUDIES/his_img_GNN_classification/pretrain_model_weights/efficientnet_v2_s-dd5fe13b.pth"))
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=False)
        model.load_state_dict(torch.load("/mnt/miaai/STUDIES/his_img_GNN_classification/pretrain_model_weights/densenet201_tv_in1k.bin"))
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == 'vit':
        model = models.vit_b_16(pretrained=False)
        model.load_state_dict(torch.load("/mnt/miaai/STUDIES/his_img_GNN_classification/pretrain_model_weights/vit_b_16-c867db91.pth"))
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # train_preds, train_labels_list = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            # preds = torch.argmax(outputs, 1).cpu().numpy()
            # train_preds.extend(preds)
            # train_labels_list.extend(labels.cpu().numpy())

        model.eval()
        val_loss = 0.0
        # val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                # preds = torch.argmax(outputs, 1).cpu().numpy()
                # val_preds.extend(preds)
                # val_labels.extend(labels.cpu().numpy())

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print('-' * 60)

    return best_model_state


train_transform, val_transform = get_transforms()
all_conf_matrices = []
test_results = {'acc': [], 'f1': [], 'balanced_acc': [], 'auc': [], 'train_times': [], 'test_times': []}
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_images, train_val_labels)):
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

    print(f'\nFold {fold + 1}/{k_folds}')
    print('=' * 50)
    print(f'Parameters: Seed={seed}, Learning Rate={lr}, Epochs={num_epochs},\n Batch_Size={batch_size}, Model={models_to_train}, Dataset={dataset_type}')
    model = initialize_model(models_to_train, num_classes)
    train_start_time = time.time()
    best_model_state = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr)
    train_time = time.time() - train_start_time

    os.makedirs(os.path.dirname(f'best_models/{models_to_train}_{dataset_type}'), exist_ok=True)
    torch.save(best_model_state, f'best_models/{models_to_train}_{dataset_type}/fold_{fold + 1}.pth')
    model.load_state_dict(best_model_state)
    print(f'Saved best model for fold {fold + 1} based on validation loss')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_preds, test_labels_list, test_probs = [], [], []
    model.eval()
    model = model.to(device)
    test_start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            iter_start = time.time()
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, 1).cpu().numpy()
            test_preds.extend(preds)
            test_labels_list.extend(labels.numpy())
            test_probs.extend(probs)
            iter_end = time.time()
            print(f"Iteration: {iter_end - iter_start:.4f} seconds")
    test_time = time.time() - test_start_time

    conf_matrix = confusion_matrix(test_labels_list, test_preds)
    test_acc = accuracy_score(test_labels_list, test_preds)
    test_f1 = f1_score(test_labels_list, test_preds, average='macro')
    test_balanced_acc = balanced_accuracy_score(test_labels_list, test_preds)

    if num_classes == 2:
        test_auc = roc_auc_score(test_labels_list, [p[1] for p in test_probs])
    else:
        test_auc = roc_auc_score(test_labels_list, test_probs, multi_class='ovr')

    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Balanced Acc: {test_balanced_acc:.4f}, Test AUC: {test_auc:.4f}, confusion_matrix:{conf_matrix}')
    print(f'Train Time: {train_time:.4f}s, Test Time: {test_time/len(test_dataset):.4f}s')

    all_conf_matrices.append(conf_matrix)
    test_results['acc'].append(test_acc)
    test_results['f1'].append(test_f1)
    test_results['balanced_acc'].append(test_balanced_acc)
    test_results['auc'].append(test_auc)
    test_results['train_times'].append(train_time)
    test_results['test_times'].append(test_time/len(test_dataset))

num_params = sum(p.numel() for p in model.parameters())
# sample_image, _ = train_dataset[0]  # a sample ?????#########
# image_memory_size = sys.getsizeof(sample_image.storage()) / (1024 * 1024)

avg_test_acc = np.mean(test_results['acc'])
std_test_acc = np.std(test_results['acc'])
avg_test_f1 = np.mean(test_results['f1'])
std_test_f1 = np.std(test_results['f1'])
avg_test_balanced_acc = np.mean(test_results['balanced_acc'])
std_test_balanced_acc = np.std(test_results['balanced_acc'])
avg_test_auc = np.mean(test_results['auc'])
std_test_auc = np.std(test_results['auc'])
avg_train_time = np.mean(test_results['train_times'])
avg_test_time = np.mean(test_results['test_times'])
std_train_time = np.std(test_results['train_times'])
std_test_time = np.std(test_results['test_times'])
sum_conf_matrix = np.sum(all_conf_matrices, axis=0)

print('\nFinal Summary Across 5 Folds:')
print('=' * 50)
print(f'Parameters: Seed={seed}, Learning Rate={lr}, Epochs={num_epochs},\n Batch Size={batch_size}, Model={models_to_train}, Number_Params={num_params}, Dataset={dataset_type}')
print(f'Confusion_Matrix: {sum_conf_matrix}')
print(f'Average Test Accuracy: {avg_test_acc:.4f} (±{std_test_acc:.4f})')
print(f'Average Test F1: {avg_test_f1:.4f} (±{std_test_f1:.4f})')
print(f'Average Test Balanced Accuracy: {avg_test_balanced_acc:.4f} (±{std_test_balanced_acc:.4f})')
print(f'Average Test AUC: {avg_test_auc:.4f} (±{std_test_auc:.4f})')
print(f'Average Train Time: {avg_train_time:.4f}s (±{std_train_time:.4f}s)')
print(f'Average Test Time: {avg_test_time:.4f}s (±{std_test_time:.4f}s)')
# print(f'Memory Usage of a processed image: {image_memory_size:.3f} MB')
