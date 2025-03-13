# extract dataset of pannuke contains malignant and benign
import numpy as np

fold_paths = [
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/images_fold1.npy',
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/images_fold2.npy',
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/images_fold3.npy'
]
label_paths = [
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/folds/fold1/types.npy',
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/folds/fold2/types.npy',
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/folds/fold3/types.npy'
]
mask_paths = [
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/folds/fold1/masks/masks.npy',
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/folds/fold2/masks/masks.npy',
    '/mnt/miaai/STUDIES/his_img_GNN_classification/datasets/pannuke/folds/fold3/masks/masks.npy'
]


def classify_and_extract_data(images, masks, types):
    malignant_images = []
    malignant_masks = []
    malignant_types = []
    malignant_labels = []

    benign_images = []
    benign_masks = []
    benign_types = []
    benign_labels = []
    number_images_cell = 0

    for idx, (image, mask, type_) in enumerate(zip(images, masks, types)):
        if len(np.unique(masks[idx, :, :, 5])) > 1:
            number_images_cell += 1
            # Extract the neoplastic cells (first column in the mask)
            neoplastic_mask = mask[:, :, 0]

            unique_neoplastic_cells = np.unique(neoplastic_mask)
            num_neoplastic_cells = len(unique_neoplastic_cells) - 1  # Subtract 1 to exclude 0

            total_cells = 0
            for i in range(5):
                cell_mask = mask[:, :, i]
                unique_cells = np.unique(cell_mask)
                total_cells += len(unique_cells) - 1  # Subtract 1 to exclude 0
            if total_cells > 0:
                neoplastic_proportion = num_neoplastic_cells / total_cells
            else:
                neoplastic_proportion = 0

            if num_neoplastic_cells >= 10 and neoplastic_proportion > 0.3:
                malignant_images.append(image)
                malignant_masks.append(mask)
                malignant_types.append(type_)
                malignant_labels.append('malignant')
            elif num_neoplastic_cells == 0:
                benign_images.append(image)
                benign_masks.append(mask)
                benign_types.append(type_)
                benign_labels.append('benign')

    malignant_images = np.array(malignant_images)
    malignant_masks = np.array(malignant_masks)
    malignant_types = np.array(malignant_types)
    malignant_labels = np.array(malignant_labels)

    benign_images = np.array(benign_images)
    benign_masks = np.array(benign_masks)
    benign_types = np.array(benign_types)
    benign_labels = np.array(benign_labels)

    return (
        malignant_images, malignant_masks, malignant_types, malignant_labels,
        benign_images, benign_masks, benign_types, benign_labels
    )


total_malignant_images = 0
total_benign_images = 0
for fold_idx, (fold_path, label_path, mask_path) in enumerate(zip(fold_paths, label_paths, mask_paths), 1):
    images = np.load(fold_path)
    types = np.load(label_path)
    masks = np.load(mask_path)

    (
        malignant_images, malignant_masks, malignant_types, malignant_labels,
        benign_images, benign_masks, benign_types, benign_labels
    ) = classify_and_extract_data(images, masks, types)

    total_malignant_images += len(malignant_images)
    total_benign_images += len(benign_images)
    print(f"\nResults for Fold {fold_idx} ({fold_path}):")
    print(f"Number of malignant images: {len(malignant_images)}")
    print(f"Number of benign images: {len(benign_images)}")

print(f"\nTotal number of malignant images: {total_malignant_images}")
print(f"Total number of benign images: {total_benign_images}")

# Save the extracted data (optional)
np.save('images/malignant_images.npy', malignant_images)
np.save('masks/malignant_masks.npy', malignant_masks)
np.save('types/malignant_types.npy', malignant_types)
np.save('labels/malignant_labels.npy', malignant_labels)

np.save('images/benign_images.npy', benign_images)
np.save('masks/benign_masks.npy', benign_masks)
np.save('types/benign_types.npy', benign_types)
np.save('labels/benign_labels.npy', benign_labels)