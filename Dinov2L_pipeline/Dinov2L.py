# Dinov2L.py
# %% Processing pipeline (train and test)
# importing necessary functionalities
import os
import pandas as pd
import random
from PIL import Image
import cv2

from copy import deepcopy

import time
import datetime

import json
from pathlib import Path

from textwrap import wrap

from typing import  Tuple, List, Dict, Any
import psutil
import GPUtil

import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split, Subset, ConcatDataset
from sklearn.model_selection import StratifiedKFold

from transformers import (
    Trainer, 
    TrainingArguments, 
    Dinov2ForImageClassification, 
    EarlyStoppingCallback
)

# Import the dataset classes from Datasets.py
from dataloader import HAM10000, Dermnet, IsicAtlas, Isic2024
from xAI import XAIVisualizer

# from torch.optim import Adam
# from torch.nn import CrossEntropyLoss
import evaluate

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# # Fix for the autocast warning
# torch.backends.cuda.enable_flash_sdp(False)  # Disable flash attention if causing issues
torch.cuda.is_available()

# Set your parameters
USE_CV = True  # Set to False to disable cross-validation
N_SPLITS = 5
TRAIN_SIZE = 0.9
VAL_SIZE = 0.1
USE_AUGMENTATION = False
NUM_AUGMENTS = 5

RANDOM_STATE = 42
early_stopping_patience = 3
num_train_epochs = 15 # the number of training epochs, e.g. 0.001

# %%
# creating variables for reusable string data (paths, model repository)
path_base = "/media/leontikos/DATA/DATA/SKIN_diseases_data/PREPARED_DATA/"
datasets = ['HAM10000', 'Dermnet', 'IsicAtlas']
types = ['train', 'test']

model_repository = r'facebook/dinov2-large'
device = torch.device("cuda")
device


#%%
def augment_dataset(dataset, num_augments_per_image=1):
    augmentations = [
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomAffine(degrees=20, shear=(10, 20)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    ]

    def histogram_equalization(image):
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        image_np[:, :, 0] = cv2.equalizeHist(image_np[:, :, 0])
        image_np = cv2.cvtColor(image_np, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(image_np)

    def add_gaussian_noise(image, mean=0, std=0.1):
        """Add Gaussian noise to an image."""
        image_np = np.array(image)
        noise = np.random.normal(mean, std, image_np.shape)
        noisy_image = np.clip(image_np + noise * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    augmented_data = []

    for item in tqdm(dataset):
        pixel_values = item['pixel_values']
        label = item['label']

        # Convert tensor to PIL image
        img = transforms.ToPILImage()(pixel_values)

        for i in range(num_augments_per_image):
            img_aug = img.copy()

            # Apply a random augmentation
            augmentation = random.choice(augmentations)
            img_aug = augmentation(img_aug)

            # Optionally apply histogram equalization
            if random.random() < 0.3:
                img_aug = histogram_equalization(img_aug)

            # Optionally add Gaussian noise
            if random.random() < 0.3:
                img_aug = add_gaussian_noise(img_aug)

            # Convert back to tensor
            pixel_values_aug = transforms.ToTensor()(img_aug)

            # Append augmented data
            augmented_data.append({'pixel_values': pixel_values_aug, 'label': label})

    return augmented_data

class AugmentedDataset(Dataset):
    def __init__(self, augmented_data):
        self.augmented_data = augmented_data

    def __len__(self):
        return len(self.augmented_data)

    def __getitem__(self, idx):
        item = self.augmented_data[idx]
        return {"pixel_values": list(item.values())[0], "label": list(item.values())[1]}

# %%
# Loading the datasets

# HAM10000
ham10000_train_full = HAM10000(
    model_repository,
    metadata_path=os.path.join(path_base, "HAM10000_verified", "HAM10000_metadata.csv"),
    images_dir=os.path.join(path_base, "HAM10000_verified", "HAM10000_images"),
    train=True
)

# Dermnet
dermnet_train_full = Dermnet(
    model_repository,
    base_path=os.path.join(path_base, "Dermnet_verified"),
    train=True
)

# ISIC Atlas
isic_atlas_train_full = IsicAtlas(
    model_repository,
    base_path=os.path.join(path_base, "ISicAtlas_verified"),
    train=True
)

# ISIC 2024
isic2024_dataset = Isic2024(
    model_repository,
    metadata_path=os.path.join(path_base, "Isic2024_verified", "train-metadata.csv"),
    images_dir=os.path.join(path_base, "Isic2024_verified", "image")
)

# Splitting datasets into train/val/test
ham10000_train, ham10000_test = random_split(ham10000_train_full, [0.9, 0.1])
ham10000_train, ham10000_val = random_split(ham10000_train, [0.9, 0.1])

dermnet_train, dermnet_test = random_split(dermnet_train_full, [0.9, 0.1])
dermnet_train, dermnet_val = random_split(dermnet_train, [0.9, 0.1])

isic_atlas_train, isic_atlas_test = random_split(isic_atlas_train_full, [0.9, 0.1])
isic_atlas_train, isic_atlas_val = random_split(isic_atlas_train, [0.9, 0.1])

isic2024_train, isic2024_test = random_split(isic2024_dataset, [0.9, 0.1])
isic2024_train, isic2024_val = random_split(isic2024_train, [0.9, 0.1])

# %%
# augmented_isic_atlas_train = augment_dataset(isic_atlas_train, num_augments_per_image=10)
# augmented_isic_atlas_train = AugmentedDataset(augmented_isic_atlas_train)
# augmented_isic_atlas_train = ConcatDataset([isic_atlas_train, augmented_isic_atlas_train])

# augmented_dermnet_clean_train = augment_dataset(dermnet_train, num_augments_per_image=5)
# augmented_dermnet_clean_train = AugmentedDataset(augmented_dermnet_clean_train)
# augmented_dermnet_clean_train = ConcatDataset([dermnet_train, augmented_dermnet_clean_train])

# augmented_ham10000_train = augment_dataset(ham10000_train, num_augments_per_image=5)
# augmented_ham10000_train = AugmentedDataset(augmented_ham10000_train)
# augmented_ham10000_train = ConcatDataset([ham10000_train, augmented_ham10000_train])

# %%
# creating a dictionary of datasets' names and their corresponding numbers of classes
n_labels = {
    'HAM10000': ham10000_train_full.num_classes,
    'Dermnet': dermnet_train_full.num_classes,
    'IsicAtlas': isic_atlas_train_full.num_classes,
    'Isic2024': len(set(isic2024_dataset.metadata['target']))
}

# ! OUPUT FOR NOW: {'HAM10000': 7, 'Dermnet': 23, 'IsicAtlas': 31, 'Isic2024': 2}

# %%
# loading Dinov2 from huggingface repository
def create_model_for_classification(num_labels):
    model = Dinov2ForImageClassification.from_pretrained(
        model_repository,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True
    )
    
    # Ensure model parameters are in float32
    model = model.float()
    return model

def test_model(trainer, test_dataset, class_names=None):
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("\nTest Set Metrics:")
    print(f"Loss: {metrics['eval_loss']:.4f}")
    print(f"Precision: {metrics['eval_precision']:.4f}")
    print(f"Recall: {metrics['eval_recall']:.4f}")
    print(f"F1 Score: {metrics['eval_f1']:.4f}")
    
    # Get predictions for detailed analysis
    trainer.model.eval()
    all_preds = []
    all_labels = []
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False
    )
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            outputs = trainer.model(batch['pixel_values'].to(trainer.model.device))
            predictions = outputs.logits.argmax(dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Convert numerical labels to string format if class_names not provided
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(set(all_labels)))]
    else:
        # Ensure class_names are strings
        class_names = [str(name) for name in class_names]
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Wrap long class names
    wrapped_class_names = ['\n'.join(wrap(str(l), width=20)) for l in class_names]
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 16))
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wrapped_class_names,
                yticklabels=wrapped_class_names)
    
    # Adjust label properties
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0, va='center')
    
    # Adjust layout to prevent label cutoff
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted', labelpad=20)
    plt.ylabel('True', labelpad=20)

    # Save confusion matrix
    plt.savefig('confusion_matrix.png')
    
    plt.tight_layout()
    plt.show()

    return metrics, all_preds, all_labels

# Optional: Function to create a more compact confusion matrix for many classes
def plot_compact_confusion_matrix(true_labels, predictions, class_names):
    """
    Creates a compact confusion matrix focusing on the most confused classes
    """
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate percentage of correct predictions for each class
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Get indices of classes with lowest accuracy
    class_accuracies = cm_percentage.diagonal()
    worst_classes = np.argsort(class_accuracies)[:10]  # Get 10 worst performing classes
    
    # Create reduced confusion matrix with worst performing classes
    cm_reduced = cm[worst_classes][:, worst_classes]
    reduced_class_names = [class_names[i] for i in worst_classes]
    wrapped_reduced_names = ['\n'.join(wrap(l, width=20)) for l in reduced_class_names]
    
    # Plot reduced confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_reduced, annot=True, fmt='d', cmap='Blues',
                xticklabels=wrapped_reduced_names,
                yticklabels=wrapped_reduced_names)
    
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0, va='center')
    
    plt.title('Confusion Matrix (10 Most Confused Classes)', pad=20)
    plt.xlabel('Predicted', labelpad=20)
    plt.ylabel('True', labelpad=20)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_compact.png', dpi=300, bbox_inches='tight')
    plt.show()


# -----------------------------------------------------------------------------------------------

# %%
# Create models with correct number of classes
Dinov2L_dermnet = create_model_for_classification(n_labels['Dermnet']).to(device)
Dinov2L_isicatlas = create_model_for_classification(n_labels['IsicAtlas']).to(device)
Dinov2L_ham10000 = create_model_for_classification(n_labels['HAM10000']).to(device)

# %%
# Define evaluation metrics, balanced splits, cross-validation, and trainer
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # Calculate metrics without zero_division parameter
    precision = precision_metric.compute(
        predictions=predictions, 
        references=labels, 
        average="macro"
    )["precision"]
    
    recall = recall_metric.compute(
        predictions=predictions, 
        references=labels, 
        average="macro"
    )["recall"]
    
    f1 = f1_metric.compute(
        predictions=predictions, 
        references=labels, 
        average="macro"
    )["f1"]
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

#%%
import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from torch.utils.data import Dataset, Subset

def save_dataset_configuration(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    model_path: str,
    use_cv: bool,
    n_splits: int = None,
    best_fold: int = None
) -> None:
    """
    Save dataset configuration and splits for future use.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model_path: Path where model is saved
        use_cv: Whether cross-validation was used
        n_splits: Number of CV splits (if use_cv=True)
        best_fold: Best performing fold (if use_cv=True)
    """
    save_dir = Path(model_path)
    config_path = save_dir / "dataset_config.json"
    splits_path = save_dir / "dataset_splits.pt"
    
    # Get the original dataset (not subset) to access label encoder
    original_dataset = train_dataset
    while hasattr(original_dataset, 'dataset'):
        original_dataset = original_dataset.dataset
    
    # Save dataset configuration
    config = {
        "use_cv": use_cv,
        "n_splits": n_splits if use_cv else None,
        "best_fold": best_fold if use_cv else None,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
    
    # Save class names from label encoder if available
    if hasattr(original_dataset, 'label_encoder'):
        config["classes"] = original_dataset.label_encoder.classes_.tolist()
    
    # Save indices for each split
    splits_dict = {
        "train_indices": get_dataset_indices(train_dataset),
        "val_indices": get_dataset_indices(val_dataset),
        "test_indices": get_dataset_indices(test_dataset)
    }
    
    # Save configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save splits
    torch.save(splits_dict, splits_path)
    
    print(f"Dataset configuration saved to: {config_path}")
    print(f"Dataset splits saved to: {splits_path}")

def get_dataset_indices(dataset: Dataset) -> List[int]:
    """Extract indices from a dataset or subset."""
    if isinstance(dataset, Subset):
        return dataset.indices
    else:
        return list(range(len(dataset)))

def load_dataset_configuration(
    model_path: str,
    full_dataset: Dataset
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:
    """
    Load dataset configuration and recreate splits.
    
    Args:
        model_path: Path where model and configuration are saved
        full_dataset: The complete dataset to split
        
    Returns:
        train_dataset: Reconstructed training dataset
        val_dataset: Reconstructed validation dataset
        test_dataset: Reconstructed test dataset
        config: Loaded configuration dictionary
    """
    save_dir = Path(model_path)
    config_path = save_dir / "dataset_config.json"
    splits_path = save_dir / "dataset_splits.pt"
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load splits
    splits_dict = torch.load(splits_path)
    
    # Reconstruct datasets
    train_dataset = Subset(full_dataset, splits_dict["train_indices"])
    val_dataset = Subset(full_dataset, splits_dict["val_indices"])
    test_dataset = Subset(full_dataset, splits_dict["test_indices"])
    
    print("Dataset configuration loaded successfully")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, config

# Example usage of saving splits in train_model function:
def save_final_splits(model_path: str, train_data, val_data, final_test_set, 
                     use_cv: bool, n_splits: int = None, best_fold: int = None):
    """Helper function to save final dataset splits after training."""
    try:
        save_dataset_configuration(
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=final_test_set,
            model_path=model_path,
            use_cv=use_cv,
            n_splits=n_splits if use_cv else None,
            best_fold=best_fold if use_cv else None
        )
    except Exception as e:
        print(f"Warning: Failed to save dataset configuration: {str(e)}")
        
# Example of loading and evaluating:
def load_and_evaluate_model(model_path: str, full_dataset: Dataset, device: str = "cuda"):
    """cuda
    Load a saved model and evaluate it on the saved test split.
    
    Args:
        model_path: Path to the saved model and configuration
        full_dataset: The complete dataset
        device: Device to run evaluation on
    """
    # Load dataset configuration
    train_dataset, val_dataset, test_dataset, config = load_dataset_configuration(
        model_path=model_path,
        full_dataset=full_dataset
    )
    
    # Load model
    model = Dinov2ForImageClassification.from_pretrained(model_path).to(device)
    
    # Create evaluation trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(model_path, "evaluation"),
        per_device_eval_batch_size=32,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    # Get detailed predictions and visualization
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False
    )
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            outputs = model(batch['pixel_values'].to(device))
            predictions = outputs.logits.argmax(dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Get class names from config if available
    class_names = config.get("classes", [f"Class_{i}" for i in range(len(set(all_labels)))])
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'test_confusion_matrix.png'))
    plt.close()
    
    # Print classification report
    print("\nTest Set Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return test_results, (all_preds, all_labels)

def create_balanced_splits(dataset, train_size: float = 0.9, val_size: float = 0.1,
                         random_state: int = 42) -> Tuple[Subset, Subset, Subset]:
    """
    Create balanced train/val/test splits ensuring proportional class representation.
    
    Args:
        dataset: PyTorch Dataset instance
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        random_state: Random seed for reproducibility
    """
    # Get all labels
    all_labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    unique_labels = np.unique(all_labels)
    
    # Create indices for each class
    class_indices = {label: [] for label in unique_labels}
    for idx, label in enumerate(all_labels):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # For each class, split indices proportionally
    for label in unique_labels:
        indices = class_indices[label]
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        n_samples = len(indices)
        n_train = int(n_samples * train_size)
        n_val = int(n_samples * val_size)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Shuffle the combined indices
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return (Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices))

def setup_cross_validation(dataset, n_splits: int = 5, random_state: int = 42):
    """
    Set up cross-validation splits while maintaining class balance.
    
    Args:
        dataset: PyTorch Dataset instance
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
    """
    # Get labels for stratification
    labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Generate and return the splits
    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        splits.append((Subset(dataset, train_idx), Subset(dataset, val_idx)))
    
    return splits

def prepare_datasets(full_dataset, test_size=0.1, random_state=42):
    """
    Prepare datasets for either CV or single split training
    """
    # Create the held-out test set first
    train_size = 1 - test_size
    train_full, test_set = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    return train_full, test_set

def train_model(model, full_dataset, labels, training_args, use_cv=False, n_splits=5,
                train_size=0.8, val_size=0.1, test_size=0.1,
                use_augmentation=True, num_augments_per_image=5,
                random_state=42):
    """
    Train model with proper dataset splitting and best model tracking
    """
    # Initialize logger
    logger = ExperimentLogger(
        base_path=training_args.output_dir,
        experiment_name=f"dinov2_training_{datetime.datetime.now().strftime('%Y%m%d')}"
    )
    
    # Log configuration
    config = create_experiment_config(training_args, "DINOv2-Large")
    config['num_parameters'] = sum(p.numel() for p in model.parameters())
    logger.log_config(config)
    
    # First, create the held-out test set
    train_full, final_test_set = prepare_datasets(
        full_dataset, 
        test_size=test_size, 
        random_state=random_state
    )
    
    if use_cv:
        # Perform cross-validation on train_full
        cv_splits = setup_cross_validation(train_full, n_splits, random_state)
        cv_results = []
        best_f1 = 0
        best_model = None
        best_model_fold = None
        best_fold_splits = None 
        
        for fold, (train_data, val_data) in enumerate(cv_splits, 1):
            logger.start_fold(fold)
            
            # Create fresh model for each fold
            fold_model = create_model_for_classification(labels).to(device)
            
            # Log split statistics
            logger.log_dataset_stats(train_data, f"Fold {fold} Training")
            logger.log_dataset_stats(val_data, f"Fold {fold} Validation")
            
            # Apply augmentation if requested
            if use_augmentation:
                augmented_train = augment_dataset(train_data, num_augments_per_image)
                augmented_train = AugmentedDataset(augmented_train)
                train_data = ConcatDataset([train_data, augmented_train])
                logger.log_message(f"Added {len(augmented_train)} augmented samples")
            
            # Create and train the model for this fold
            trainer = Trainer(
                model=fold_model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            )
            
            # Train and evaluate
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            # Log fold results
            fold_metrics = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_precision': eval_result['eval_precision'],
                'eval_recall': eval_result['eval_recall'],
                'eval_f1': eval_result['eval_f1']
            }
            
            # Track best model
            if eval_result['eval_f1'] > best_f1:
                best_f1 = eval_result['eval_f1']
                best_model = deepcopy(trainer.model)
                best_model_fold = fold
                # Store the splits from the best fold
                best_fold_splits = {
                    'train_data': train_data,
                    'val_data': val_data
                }
                
            # Get confusion matrix for this fold
            trainer.model.eval()
            all_preds = []
            all_labels = []
            val_dataloader = torch.utils.data.DataLoader(
                val_data, batch_size=training_args.per_device_eval_batch_size
            )
            
            with torch.no_grad():
                for batch in val_dataloader:
                    outputs = trainer.model(batch['pixel_values'].to(device))
                    predictions = outputs.logits.argmax(dim=-1)
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())
            
            cm = confusion_matrix(all_labels, all_preds)
            additional_info = {
                'confusion_matrix': cm.tolist(),
                'best_model_checkpoint': trainer.state.best_model_checkpoint
            }
            
            logger.end_fold(fold_metrics, additional_info)
            cv_results.append(fold_metrics)
        
        logger.log_message(f"\nBest model was from fold {best_model_fold} with F1 score: {best_f1:.4f}")
        
        # Evaluate best model on held-out test set
        logger.log_message("\nEvaluating best model on held-out test set...")
        # Create new training arguments for testing (without evaluation strategy)
        test_args = deepcopy(training_args)
        test_args.eval_strategy = "no"
        
        test_trainer = Trainer(
            model=best_model,
            args=test_args,
            compute_metrics=compute_metrics,
            eval_dataset=final_test_set  # Provide the eval_dataset here
        )
        final_test_results = test_trainer.evaluate()
        logger.log_message("Final Test Set Results:")
        logger.log_metrics(final_test_results)
        
        # Save best model
        best_model_path = os.path.join(training_args.output_dir, "best_model")
        best_model.save_pretrained(best_model_path)
        logger.log_message(f"Best model saved to: {best_model_path}")

        # Save the final splits from the best fold
        save_final_splits(
            model_path=best_model_path,
            train_data=train_data,  # from best fold
            val_data=val_data,      # from best fold
            final_test_set=final_test_set,
            use_cv=True,
            n_splits=n_splits,
            best_fold=best_model_fold
        )
        
        return logger, cv_results, final_test_results, best_model
        
    else:
        # For non-CV training, use balanced splits
        train_data, val_data, final_test_set = create_balanced_splits(
            full_dataset,
            train_size=train_size,
            val_size=val_size,
            random_state=random_state
        )
        
        # Log split statistics
        logger.log_dataset_stats(train_data, "Training")
        logger.log_dataset_stats(val_data, "Validation")
        logger.log_dataset_stats(final_test_set, "Test")
        
        if use_augmentation:
            augmented_train = augment_dataset(train_data, num_augments_per_image)
            augmented_train = AugmentedDataset(augmented_train)
            train_data = ConcatDataset([train_data, augmented_train])
            logger.log_message(f"Added {len(augmented_train)} augmented samples")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train and evaluate
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        
        # Evaluate on held-out test set
        test_results = trainer.evaluate(eval_dataset=final_test_set)
        logger.log_message("Final Test Set Results:")
        logger.log_metrics(test_results)
        
        # Save best model
        model_path = os.path.join(training_args.output_dir, "best_model")
        trainer.model.save_pretrained(model_path)
        logger.log_message(f"Model saved to: {model_path}")

        # Save the final splits
        save_final_splits(
            model_path=model_path,
            train_data=train_data,
            val_data=val_data,
            final_test_set=final_test_set,
            use_cv=False
        )
        
        return trainer, train_result, eval_result, test_results, logger

class ExperimentLogger:
    def __init__(self, base_path: str, experiment_name: str):
        """
        Initialize experiment logger
        
        Args:
            base_path: Base directory for saving experiment logs
            experiment_name: Name of the experiment
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.base_path / f"{experiment_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.fold_metrics = []
        self.training_start_time = None
        
        # Create log files
        self.main_log_file = self.exp_dir / "experiment_log.txt"
        self.metrics_file = self.exp_dir / "metrics.json"
        self.config_file = self.exp_dir / "config.json"
        
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """
        Log metrics to the main log file and save them to metrics.json
        
        Args:
            metrics: Dictionary of metric names and values
            prefix: Optional prefix for metric names in the log
        """
        self.log_message("\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.log_message(f"{prefix}{key}: {value:.4f}")
            else:
                self.log_message(f"{prefix}{key}: {value}")
        
        # Format metrics with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_metrics = {
            'timestamp': timestamp,
            'metrics': {f"{prefix}{k}": v for k, v in metrics.items()}
        }
        
        # Append to fold metrics list
        self.fold_metrics.append(new_metrics)
        
        # Save updated metrics
        self._save_metrics()
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Also log to main log file
        self.log_message("\nExperiment Configuration:")
        for key, value in config.items():
            self.log_message(f"{key}: {value}")
    
    def log_dataset_stats(self, dataset, split_name: str):
        """Log dataset statistics"""
        stats = {
            'total_samples': len(dataset),
            'num_classes': dataset.num_classes if hasattr(dataset, 'num_classes') else None,
            'class_distribution': self._get_class_distribution(dataset)
        }
        
        self.log_message(f"\n{split_name} Dataset Statistics:")
        self.log_message(f"Total samples: {stats['total_samples']}")
        if stats['num_classes']:
            self.log_message(f"Number of classes: {stats['num_classes']}")
        self.log_message("Class distribution:")
        for class_name, count in stats['class_distribution'].items():
            self.log_message(f"  {class_name}: {count}")
            
        # Save distribution plot
        self._plot_class_distribution(stats['class_distribution'], split_name)
    
    def start_fold(self, fold_num: int):
        """Log the start of a new fold"""
        self.training_start_time = time.time()
        self.current_fold = fold_num
        self.log_message(f"\nStarting Fold {fold_num}")
        self._log_system_state()
    
    def end_fold(self, metrics: Dict[str, float], additional_info: Dict[str, Any] = None):
        """Log the end of a fold"""
        training_time = time.time() - self.training_start_time
        
        # Combine metrics with timing information
        fold_results = {
            'fold': self.current_fold,
            'training_time': training_time,
            **metrics
        }
        if additional_info:
            fold_results.update(additional_info)
        
        self.fold_metrics.append(fold_results)
        
        # Log to main log file
        self.log_message(f"\nFold {self.current_fold} Results:")
        for key, value in fold_results.items():
            if isinstance(value, float):
                self.log_message(f"{key}: {value:.4f}")
            else:
                self.log_message(f"{key}: {value}")
        
        # Save updated metrics
        self._save_metrics()
    
    def log_final_results(self, confusion_mat=None, class_names=None):
        """Log final experimental results"""
        # Calculate aggregate statistics
        mean_metrics = {}
        std_metrics = {}
        
        for metric in self.fold_metrics[0].keys():
            if metric not in ['fold', 'training_time']:
                values = [fold[metric] for fold in self.fold_metrics]
                mean_metrics[metric] = np.mean(values)
                std_metrics[metric] = np.std(values)
        
        self.log_message("\nFinal Results:")
        for metric, mean_value in mean_metrics.items():
            std_value = std_metrics[metric]
            self.log_message(f"Mean {metric}: {mean_value:.4f} ± {std_value:.4f}")
        
        # Save confusion matrix if provided
        if confusion_mat is not None and class_names is not None:
            self._plot_confusion_matrix(confusion_mat, class_names)
    
    def log_message(self, message: str):
        """Log a message to the main log file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.main_log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def _log_system_state(self):
        """Log system resource usage"""
        gpu_usage = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        
        system_state = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_used': f"{gpu_usage.memoryUsed}MB" if gpu_usage else "N/A",
            'gpu_utilization': f"{gpu_usage.load*100}%" if gpu_usage else "N/A"
        }
        
        self.log_message("\nSystem State:")
        for key, value in system_state.items():
            self.log_message(f"{key}: {value}")
    
    def _get_class_distribution(self, dataset) -> Dict[str, int]:
        """Calculate class distribution"""
        if hasattr(dataset, 'get_class_distribution'):
            return dataset.get_class_distribution()
        
        # Default implementation for datasets without the method
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(map(str, unique), counts))
    
    def _plot_class_distribution(self, distribution: Dict[str, int], split_name: str):
        """Create and save class distribution plot"""
        plt.figure(figsize=(12, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.title(f"Class Distribution - {split_name}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.exp_dir / f"class_distribution_{split_name}.png")
        plt.close()
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Create and save confusion matrix plot"""
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.exp_dir / "confusion_matrix.png")
        plt.close()
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.fold_metrics, f, indent=4)

def create_experiment_config(args, model_name: str) -> Dict[str, Any]:
    """Create a configuration dictionary for the experiment"""
    return {
        'model_name': model_name,
        'num_epochs': args.num_train_epochs,
        'batch_size': args.per_device_train_batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'fp16': args.fp16,
        'gradient_checkpointing': args.gradient_checkpointing,
        'max_grad_norm': args.max_grad_norm,
        'optimizer': args.optim,
        'num_parameters': None  # To be filled with actual model parameters
    }

def log_model_info(model, logger: ExperimentLogger):
    """Log model architecture information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.log_message("\nModel Information:")
    logger.log_message(f"Total parameters: {total_params:,}")
    logger.log_message(f"Trainable parameters: {trainable_params:,}")
    logger.log_message(f"Architecture:\n{str(model)}")


'''def get_dermnet_class_names(base_path):
    """
    Extract class names from Dermnet dataset folder structure.
    
    Args:
        base_path: Base path to the Dermnet dataset directory
    Returns:
        list: List of class names
    """
    base_path = Path(base_path)
    train_dir = base_path / "train"
    
    if not train_dir.exists():
        print(f"Warning: Could not find train directory at {train_dir}")
        return None
        
    # Get class names from directory names
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return class_names'''

def get_dataset_class_names(base_path):
    """
    Extract class names from one of the three datasets.
    
    Args:
        base_path: Base path to the chosen dataset directory
    Returns:
        list: List of class names
    """
    base_path = Path(base_path)
    train_dir = base_path / "train"
    metadata = base_path / "metadata.csv"
    
    # Get class names based on dataset folder structure
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_names
        
    elif metadata.exists():
        class_names = sorted(pd.read_csv('your_file.csv')['dx'].unique())
        return class_names
    else:
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        return class_names

def generate_confusion_matrix_with_percentages(trainer, test_dataset, save_dir, full_dataset=None):
    """Generate and save confusion matrix with both counts and percentages."""
    trainer.model.eval()
    all_preds = []
    all_labels = []
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False
    )
    
    print("Getting predictions...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            outputs = trainer.model(batch['pixel_values'].to(trainer.model.device))
            predictions = outputs.logits.argmax(dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Try to get dataset class names
    class_names = None
    if hasattr(full_dataset, 'base_path'):
        class_names = get_dataset_class_names(full_dataset.base_path)
    
    # Fallback to other sources if dataset class names not found
    if class_names is None:
        class_names = get_class_names(Path(save_dir), full_dataset)
    
    if class_names is None:
        print("Warning: Could not find class names, using generic labels")
        class_names = [f"Class_{i}" for i in range(len(set(all_labels)))]
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure (make it larger for combined display)
    plt.figure(figsize=(25, 20))
    
    # Create annotation text with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            if not np.isnan(percentage):
                annotations[i, j] = f'{count}\n({percentage:.1f}%)'
            else:
                annotations[i, j] = f'{count}\n(0.0%)'
    
    # Plot confusion matrix
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Test Set Confusion Matrix\n(Count and Percentage)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the confusion matrix
    save_path = Path(save_dir) / 'confusion_matrix_with_percentages.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print classification report with proper labels
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Print per-class accuracy with class names
    print("\nPer-class Accuracy:")
    for i in range(len(class_names)):
        class_mask = (np.array(all_labels) == i)
        if np.any(class_mask):
            class_acc = np.mean(np.array(all_preds)[class_mask] == i)
            total_samples = np.sum(class_mask)
            correct_samples = np.sum(np.array(all_preds)[class_mask] == i)
            print(f"{class_names[i]}:")
            print(f"  Accuracy: {class_acc:.4f}")
            print(f"  Correct predictions: {correct_samples}/{total_samples}")

def load_and_evaluate_saved_model(model_path: str, full_dataset=None, device='cuda'):
    """
    Load a saved model and evaluate it on the original test split.
    
    Args:
        model_path: Path to the saved model
        full_dataset: The full dataset object used during training
        device: Device to run the model on
    """
    try:
        model_dir = Path(model_path)
        
        # Load the model
        model = Dinov2ForImageClassification.from_pretrained(model_path).to(device)
        print("Model loaded successfully")
        
        # Load dataset configuration to get the original splits
        if full_dataset is not None and (model_dir / "dataset_splits.pt").exists():
            print("Loading original dataset splits...")
            splits_dict = torch.load(model_dir / "dataset_splits.pt")
            
            # Reconstruct test dataset using original indices
            test_dataset = Subset(full_dataset, splits_dict["test_indices"])
            print(f"Reconstructed test set with {len(test_dataset)} samples")
            
            # Create evaluation trainer
            training_args = TrainingArguments(
                output_dir=os.path.join(model_path, "evaluation"),
                per_device_eval_batch_size=32,
                remove_unused_columns=False,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics
            )
            
            # Evaluate on test dataset
            print("\nEvaluating model on original test split...")
            test_results = trainer.evaluate(test_dataset)
            
            # Print test results
            print("\nTest Set Results:")
            print("-"*40)
            for metric, value in test_results.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
            
            # Generate and save enhanced confusion matrix
            print("\nGenerating confusion matrix with percentages...")
            generate_confusion_matrix_with_percentages(trainer, test_dataset, model_path, full_dataset)
            
            return model, test_results
        else:
            print("Warning: Could not find original dataset splits or full dataset not provided")
            return model, None
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def get_class_names(model_dir, dataset=None):
    """
    Get the proper class names from available sources.
    
    Args:
        model_dir: Path to the model directory
        dataset: The dataset object that was used for training
    
    Returns:
        list: List of class names
    """
    # First try to get from dataset's label encoder
    if dataset is not None and hasattr(dataset, 'label_encoder'):
        return dataset.label_encoder.classes_
    
    # Then try to get class names from dataset_config.json
    try:
        if (model_dir / "dataset_config.json").exists():
            with open(model_dir / "dataset_config.json", "r") as f:
                dataset_config = json.load(f)
                if "classes" in dataset_config:
                    return dataset_config["classes"]
    except Exception as e:
        print(f"Warning: Could not load class names from dataset_config.json: {e}")
    
    # If all else fails, try to get from the model's config.json
    try:
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)
            if "id2label" in config:
                num_labels = len(config["id2label"])
                print("Warning: Using generic labels from model config")
                return [config["id2label"][str(i)] for i in range(num_labels)]
    except Exception as e:
        print(f"Warning: Could not load class names from config.json: {e}")
    
    return None

# %%
# defining the trainer and training the model (for DermNet)
training_args = TrainingArguments(
    output_dir=os.path.join(path_base, "OUTPUT", "dinov2L_dermnet_CV"),
    eval_strategy="epoch",
    save_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # Increased from 32 to 128
    per_device_eval_batch_size=32,   # Increased from 32 to 128
    num_train_epochs=num_train_epochs,
    optim="adamw_torch",
    gradient_accumulation_steps=1,    # Used for increased batch size
    warmup_ratio=0.1,                 # Added to prevent overfitting
    weight_decay=0.01,
    logging_dir=os.path.join(path_base, "OUTPUT", "dinov2L_dermnet_CV", "logs"),
    logging_steps=10,
    fp16=True,                        # Enable mixed precision training
    gradient_checkpointing=True,      # Memory efficient training
    max_grad_norm=1.0,                # Help stabilize training
    dataloader_pin_memory=True,       # Faster data transfer to GPU
    load_best_model_at_end=True,
    report_to="tensorboard",          # Better training monitoring
    metric_for_best_model="f1",
    greater_is_better=True,           # Better training monitoring
    save_total_limit=1,  # Keep only the last 1 checkpoint(s)
)

# Print training info
print(f"Number of training examples: {len(dermnet_train)}")
print(f"Number of validation examples: {len(dermnet_val)}")
print(f"Number of classes: {n_labels['Dermnet']}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Number of epochs: {training_args.num_train_epochs}")

# Training call with logging
if USE_CV:
    logger, cv_results, final_test_results, best_model = train_model(
        model=Dinov2L_dermnet,
        full_dataset=dermnet_train_full,  # Use the full dataset
        labels=n_labels['Dermnet'],
        training_args=training_args,
        use_cv=True,
        n_splits=N_SPLITS,
        test_size=0.1,  # Explicit test set size
        use_augmentation=USE_AUGMENTATION,
        num_augments_per_image=NUM_AUGMENTS,
        random_state=RANDOM_STATE
    )
    print(f"\nExperiment completed. Check the logs at: {logger.exp_dir}")
else:
    trainer, train_result, eval_result, test_results, logger = train_model(
        model=Dinov2L_dermnet,
        full_dataset=dermnet_train_full,  # Use the full dataset
        labels=n_labels['Dermnet'],
        training_args=training_args,
        use_cv=False,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        use_augmentation=USE_AUGMENTATION,
        num_augments_per_image=NUM_AUGMENTS,
        random_state=RANDOM_STATE
    )
    print(f"\nExperiment completed. Check the logs at: {logger.exp_dir}")


# Print the final test results
print("\nFinal Test Results:")
print(final_test_results)

# %%
# Show the results and visualizations on the test set

# Path to the saved model
model_path = os.path.join(path_base, "OUTPUT", "dinov2L_dermnet_CV", "best_model")

# Load and evaluate using the original dataset to ensure proper splits and labels
model, test_results = load_and_evaluate_saved_model(
    model_path,
    full_dataset=dermnet_train_full,  # Pass the full dataset for proper reconstruction
    device='cuda'
)

# %%
# xAI Visualizer

# Initialize the XAI visualizer with your best model
xai = XAIVisualizer(model, device='cuda')

# Get class names
if hasattr(dermnet_train_full, 'label_encoder'):
    class_names = dermnet_train_full.label_encoder.classes_.tolist()
else:
    class_names = [f"Class_{i}" for i in range(n_labels['Dermnet'])]

# Create directory for saving visualizations
save_dir = os.path.join(path_base, "OUTPUT", "dinov2L_dermnet_CV", "xai_visualizations")
os.makedirs(save_dir, exist_ok=True)

# Generate visualizations
xai.analyze_predictions(
    dataset=dermnet_test,
    class_names=class_names,
    n_correct=10,
    n_incorrect=10,
    save_dir=save_dir
)

# %%
# defining the trainer and training the model (for HAM10000)
training_args = TrainingArguments(
    output_dir=os.path.join(path_base, "OUTPUT", "dinov2L_ham10000_CV"),
    eval_strategy="epoch",
    save_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # Increased from 32 to 128
    per_device_eval_batch_size=32,   # Increased from 32 to 128
    num_train_epochs=num_train_epochs,
    optim="adamw_torch",
    gradient_accumulation_steps=1,    # Used for increased batch size
    warmup_ratio=0.1,                 # Added to prevent overfitting
    weight_decay=0.01,
    logging_dir=os.path.join(path_base, "OUTPUT", "dinov2L_ham10000_CV", "logs"),
    logging_steps=10,
    fp16=True,                        # Enable mixed precision training
    gradient_checkpointing=True,      # Memory efficient training
    max_grad_norm=1.0,                # Help stabilize training
    dataloader_pin_memory=True,       # Faster data transfer to GPU
    load_best_model_at_end=True,
    report_to="tensorboard",          # Better training monitoring
    metric_for_best_model="f1",
    greater_is_better=True,           # Better training monitoring
    save_total_limit=1,  # Keep only the last 1 checkpoint(s)
)

# Print training info
print(f"Number of training examples: {len(ham10000_train)}")
print(f"Number of validation examples: {len(ham10000_val)}")
print(f"Number of classes: {n_labels['HAM10000']}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Number of epochs: {training_args.num_train_epochs}")

# Training call with logging
if USE_CV:
    logger, cv_results, final_test_results, best_model = train_model(
        model=Dinov2L_ham10000,
        full_dataset=ham10000_train_full,  # Use the full dataset
        labels=n_labels['HAM10000'],
        training_args=training_args,
        use_cv=True,
        n_splits=N_SPLITS,
        test_size=0.1,  # Explicit test set size
        use_augmentation=USE_AUGMENTATION,
        num_augments_per_image=NUM_AUGMENTS,
        random_state=RANDOM_STATE
    )
    print(f"\nExperiment completed. Check the logs at: {logger.exp_dir}")
else:
    trainer, train_result, eval_result, test_results, logger = train_model(
        model=Dinov2L_ham10000,
        full_dataset=ham10000_train_full,  # Use the full dataset
        labels=n_labels['HAM10000'],
        training_args=training_args,
        use_cv=False,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        use_augmentation=USE_AUGMENTATION,
        num_augments_per_image=NUM_AUGMENTS,
        random_state=RANDOM_STATE
    )
    print(f"\nExperiment completed. Check the logs at: {logger.exp_dir}")


# Print the final test results
print("\nFinal Test Results:")
print(final_test_results)

# %%
# Show the results and visualizations on the test set

# Path to the saved model
model_path = os.path.join(path_base, "OUTPUT", "dinov2L_ham10000_CV", "best_model")

# Load and evaluate using the original dataset to ensure proper splits and labels
model, test_results = load_and_evaluate_saved_model(
    model_path,
    full_dataset=ham10000_train_full,  # Pass the full dataset for proper reconstruction
    device='cuda'
)

# %%
# xAI Visualizer

# Initialize the XAI visualizer with your best model
xai = XAIVisualizer(model, device='cuda')

# Get class names
if hasattr(ham10000_train_full, 'label_encoder'):
    class_names = ham10000_train_full.label_encoder.classes_.tolist()
else:
    class_names = [f"Class_{i}" for i in range(n_labels['HAM10000'])]

# Create directory for saving visualizations
save_dir = os.path.join(path_base, "OUTPUT", "dinov2L_ham10000_CV", "xai_visualizations")
os.makedirs(save_dir, exist_ok=True)

# Generate visualizations
xai.analyze_predictions(
    dataset=ham10000_test,
    class_names=class_names,
    n_correct=10,
    n_incorrect=10,
    save_dir=save_dir
)

# %%
# defining the trainer and training the model (for IsicAtlas)
training_args = TrainingArguments(
    output_dir=os.path.join(path_base, "OUTPUT", "dinov2L_isicatlas_CV"),
    eval_strategy="epoch",
    save_strategy="epoch",  
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # Increased from 32 to 128
    per_device_eval_batch_size=32,   # Increased from 32 to 128
    num_train_epochs=num_train_epochs,
    optim="adamw_torch",
    gradient_accumulation_steps=1,    # Used for increased batch size
    warmup_ratio=0.1,                 # Added to prevent overfitting
    weight_decay=0.01,
    logging_dir=os.path.join(path_base, "OUTPUT", "dinov2L_isicatlas_CV", "logs"),
    logging_steps=10,
    fp16=True,                        # Enable mixed precision training
    gradient_checkpointing=True,      # Memory efficient training
    max_grad_norm=1.0,                # Help stabilize training
    dataloader_pin_memory=True,       # Faster data transfer to GPU
    load_best_model_at_end=True,
    report_to="tensorboard",          # Better training monitoring
    metric_for_best_model="f1",
    greater_is_better=True,           # Better training monitoring
    save_total_limit=1,  # Keep only the last 1 checkpoint(s)
)

# Print training info
print(f"Number of training examples: {len(isic_atlas_train)}")
print(f"Number of validation examples: {len(isic_atlas_val)}")
print(f"Number of classes: {n_labels['IsicAtlas']}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Number of epochs: {training_args.num_train_epochs}")

# Training call with logging
if USE_CV:
    logger, cv_results, final_test_results, best_model = train_model(
        model=Dinov2L_isicatlas,
        full_dataset=isic_atlas_train_full,  # Use the full dataset
        labels=n_labels['IsicAtlas'],
        training_args=training_args,
        use_cv=True,
        n_splits=N_SPLITS,
        test_size=0.1,  # Explicit test set size
        use_augmentation=USE_AUGMENTATION,
        num_augments_per_image=NUM_AUGMENTS,
        random_state=RANDOM_STATE
    )
    print(f"\nExperiment completed. Check the logs at: {logger.exp_dir}")
else:
    trainer, train_result, eval_result, test_results, logger = train_model(
        model=Dinov2L_isicatlas,
        full_dataset=isic_atlas_train_full,  # Use the full dataset
        labels=n_labels['IsicAtlas'],
        training_args=training_args,
        use_cv=False,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        use_augmentation=USE_AUGMENTATION,
        num_augments_per_image=NUM_AUGMENTS,
        random_state=RANDOM_STATE
    )
    print(f"\nExperiment completed. Check the logs at: {logger.exp_dir}")


# Print the final test results
print("\nFinal Test Results:")
print(final_test_results)

# %%
# Show the results and visualizations on the test set

# Path to the saved model
model_path = os.path.join(path_base, "OUTPUT", "dinov2L_isicatlas_CV", "best_model")

# Load and evaluate using the original dataset to ensure proper splits and labels
model, test_results = load_and_evaluate_saved_model(
    model_path,
    full_dataset=isic_atlas_train_full,  # Pass the full dataset for proper reconstruction
    device='cuda'
)

# %%
# xAI Visualizer

# Initialize the XAI visualizer with your best model
xai = XAIVisualizer(model, device='cuda')

# Get class names
if hasattr(isic_atlas_train_full, 'label_encoder'):
    class_names = isic_atlas_train_full.label_encoder.classes_.tolist()
else:
    class_names = [f"Class_{i}" for i in range(n_labels['IsicAtlas'])]

# Create directory for saving visualizations
save_dir = os.path.join(path_base, "OUTPUT", "dinov2L_isicatlas_CV", "xai_visualizations")
os.makedirs(save_dir, exist_ok=True)

# Generate visualizations
xai.analyze_predictions(
    dataset=isic_atlas_test,
    class_names=class_names,
    n_correct=10,
    n_incorrect=10,
    save_dir=save_dir
)


# %%
