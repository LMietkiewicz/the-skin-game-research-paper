# The Skin Game: AI Dermatology Model Evaluation Framework

This repository contains the implementation code for the paper "The Skin Game: Revolutionizing Standards for AI Dermatology Model Comparison". The framework provides a comprehensive approach to training and evaluating deep learning models for dermatological image classification, with a particular focus on methodology standardization and reproducibility.

## Overview

The project implements a robust evaluation framework for skin disease classification using the DINOv2-Large vision transformer model. It includes:

- Standardized data preprocessing and augmentation pipelines
- Cross-validation implementation with proper data splitting
- Comprehensive performance metrics and visualization tools
- Explainable AI functionality for model interpretation
- Support for multiple benchmark datasets (HAM10000, DermNet, ISIC Atlas)

## Requirements

### Hardware Requirements
- GPU with at least 16GB VRAM (tested on NVIDIA GeForce RTX 3090 24GB VRAM)
- 32GB RAM recommended
- Ubuntu 22.04.5 LTS or compatible OS (Windows / Mac)

### Software Requirements
```
Python 3.9+
PyTorch 2.0+
torchvision
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
opencv-python
tqdm
```

## Project Structure

```
├── Dinov2L_pipeline    # Folder containing the training, test, and evaluation pipeline
│   ├── Dinov2L.py      # Main training and evaluation pipeline
│   ├── dataloader.py   # Dataset classes and data loading utilities
│   └── xAI.py          # Explainable AI visualization tools
├── LICENSE.md          # Project license
└── README.md           # This file
```

## Dataset Preparation

The framework supports three main datasets:

1. HAM10000 (7 classes)
2. DermNet (23 classes)
3. ISIC Atlas (31 classes)

Each dataset should be organized in the following structure:

```
PREPARED_DATA/
├── HAM10000_verified/
│   ├── HAM10000_metadata.csv
│   └── HAM10000_images/
├── Dermnet_verified/
│   └── [class_folders]/
└── ISicAtlas_verified/
    └── [class_folders]/
```

## Usage

### Training a Model

```python
from Dinov2L import train_model, TrainingArguments

# Configure training arguments
training_args = TrainingArguments(
    output_dir="OUTPUT/dinov2L_experiment",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=15,
    # See code for additional parameters
)

# Train with cross-validation
logger, cv_results, final_test_results, best_model = train_model(
    model=model,
    full_dataset=dataset,
    labels=n_labels,
    training_args=training_args,
    use_cv=True,
    n_splits=5
)
```

### Generating Visualizations

```python
from xAI import XAIVisualizer

# Initialize visualizer
xai = XAIVisualizer(model, device='cuda')

# Generate attention visualizations
xai.analyze_predictions(
    dataset=test_dataset,
    class_names=class_names,
    n_correct=10,
    n_incorrect=10,
    save_dir="visualization_output"
)
```

## Model Performance

The framework achieves the following macro-averaged F1-scores on the test sets:

- HAM10000: 0.85
- DermNet: 0.71
- ISIC Atlas: 0.84

## Features

- **Robust Cross-Validation**: Implements 5-fold cross-validation with proper stratification
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Mixed Precision Training**: Supports FP16 training for improved efficiency
- **Comprehensive Logging**: Detailed training logs and performance metrics
- **Attention Visualization**: Tools for understanding model decision-making
- **Flexible Data Augmentation**: Configurable augmentation pipeline

## Citing

If you use this code in your research, please cite our paper:

```bibtex
@ARTICLE{Mietkiewicz2025-av,
  title         = "The skin game: Revolutionizing standards for {AI} dermatology
                   model comparison",
  author        = "Miętkiewicz, Łukasz and Ciechanowski, Leon and Jemielniak,
                   Dariusz",
  journal       = "arXiv [eess.IV]",
  abstract      = "Deep Learning approaches in dermatological image
                   classification have shown promising results, yet the field
                   faces significant methodological challenges that impede
                   proper evaluation. This paper presents a dual contribution:
                   first, a systematic analysis of current methodological
                   practices in skin disease classification research, revealing
                   substantial inconsistencies in data preparation, augmentation
                   strategies, and performance reporting; second, a
                   comprehensive training and evaluation framework demonstrated
                   through experiments with the DINOv2-Large vision transformer
                   across three benchmark datasets (HAM10000, DermNet, ISIC
                   Atlas). The analysis identifies concerning patterns,
                   including pre-split data augmentation and validation-based
                   reporting, potentially leading to overestimated metrics,
                   while highlighting the lack of unified methodology standards.
                   The experimental results demonstrate DINOv2's performance in
                   skin disease classification, achieving macro-averaged
                   F1-scores of 0.85 (HAM10000), 0.71 (DermNet), and 0.84 (ISIC
                   Atlas). Attention map analysis reveals critical patterns in
                   the model's decision-making, showing sophisticated feature
                   recognition in typical presentations but significant
                   vulnerabilities with atypical cases and composite images. Our
                   findings highlight the need for standardized evaluation
                   protocols and careful implementation strategies in clinical
                   settings. We propose comprehensive methodological
                   recommendations for model development, evaluation, and
                   clinical deployment, emphasizing rigorous data preparation,
                   systematic error analysis, and specialized protocols for
                   different image types. To promote reproducibility, we provide
                   our implementation code through GitHub. This work establishes
                   a foundation for rigorous evaluation standards in
                   dermatological image classification and provides insights for
                   responsible AI implementation in clinical dermatology.",
  month         =  feb,
  year          =  2025,
  archivePrefix = "arXiv",
  primaryClass  = "eess.IV"
}
```

## License

This project is licensed under the CC BY-NC-SA 4.0: Creative Commons Attribution-Noncommercial-ShareAlike license - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
