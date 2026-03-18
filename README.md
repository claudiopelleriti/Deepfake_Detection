# Real vs AI-Generated Faces Classification

This repository contains a Computer Vision project focused on binary face classification to distinguish between real human faces and AI-generated deepfakes. 

## Dataset
The models were trained and evaluated on custom splits combining two major data sources:
* **Real Faces**: FFHQ (Flickr-Faces-HQ) dataset.
* **Fake/AI-Generated Faces**: AI-Generated Faces dataset from Kaggle.

Two dataset scales were used during development: a smaller balanced subset of 20,000 images for rapid architectural prototyping, and a full dataset of 100,000 images for the final training pipeline.

## Model Architecture
The core of the classification engine relies on Transfer Learning. After comparing EfficientNetB0 and ResNet50, **ResNet50** (pre-trained on ImageNet-1K) was selected as the backbone due to its robustness and stability during training.

To optimize the recognition of salient facial features, a custom **Attention Layer** was integrated into the pipeline. The final architecture consists of:
* ResNet50 backbone (fine-tuned)
* Custom Attention Layer
* Global Average Pooling 2D
* Dense classification head with Dropout and Batch Normalization

Data augmentation techniques (Random Flip, Rotation, and Zoom) were applied at the input level to prevent overfitting and improve generalization on unseen data.

## Explainability and Anti-Spoofing
While the custom attention layer provided a slight quantitative improvement in standard classification metrics, its primary contribution is qualitative. The layer acts as a powerful explainability tool by weighting the feature maps extracted by ResNet50. This allows for the visualization of which specific facial regions the model focuses on to determine whether a face is real or generated, making the anti-spoofing system highly transparent and justifiable.

## Usage
The main script `assignment3.py` provides a structured interface to interact with the pipeline. It supports different execution modes:

* **TRAIN**: Trains the model on the selected dataset size (20k or 100k images), with the option to enable or disable the custom attention module.
* **TEST**: Evaluates a trained model on the dedicated test set, computing accuracy, precision, recall, f1-score, and generating the confusion matrix.
* **CLASSIFY**: Runs direct inference on custom image paths, outputting the predicted class (Real or Generated) along with the confidence score.

## Author
Claudio Pelleriti
