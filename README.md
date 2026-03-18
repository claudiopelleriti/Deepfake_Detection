# Real vs AI-Generated Faces Classification

This repository contains a Computer Vision project focused on binary classification to distinguish between real human faces and AI-generated deepfakes.

## Dataset
The models were trained and evaluated on custom splits combining two major data sources:
* Real Faces: FFHQ (Flickr-Faces-HQ) dataset.
* Fake/AI-Generated Faces: AI-Generated Faces dataset from Kaggle.

Two dataset scales were used during development: a smaller balanced subset of 20,000 images for rapid architectural prototyping, and a full dataset extended to 100,000 images for the final training pipeline. The data was systematically shuffled and split into 70% for training, 15% for validation, and 15% for testing.

## Model Architecture
The core of the classification engine relies on Transfer Learning using the **EfficientNetB0** architecture. This model, pre-trained on ImageNet-1K, was selected over ResNet50 due to its superior efficiency, better generalization, and optimal balance of depth, width, and resolution via Compound Scaling. 

The final architecture consists of:
* EfficientNetB0 backbone (with fine-tuning applied only to the last 10 layers).
* Custom Attention Layer positioned before the classifier.
* Global Average Pooling 2D.
* Dense layer (128 units) with Batch Normalization and Dropout (0.5).
* Output Dense layer with softmax activation for 2 classes.

To prevent overfitting and improve model robustness, data augmentation techniques were applied randomly during training, including horizontal flip, rotation (±10°), zoom (±10%), and contrast variation (±10%).

## Explainability
While the custom attention layer provided a slight quantitative improvement in standard classification metrics, its primary contribution is qualitative. It allows for the visualization of which specific features are weighted more heavily by the model, helping to justify the network's decisions and making the system significantly more transparent.

## Usage
The main script `assignment3.py` provides a structured command-line interface to interact with the pipeline. It supports the following modes:

* **build_model**: Constructs and saves the architecture summaries and plot diagrams for the models with and without the attention layer.
* **train**: Trains the model on the selected dataset, with optional arguments to use the 100k dataset (`use_large_dataset`) and to enable the custom attention module (`use_attention`).
* **test**: Evaluates the trained model on the dedicated test set, computing accuracy, precision, recall, f1-score, and generating the confusion matrix.
* **classify**: Runs direct inference on one or more custom image paths, outputting the predicted class (Reale or Generato) along with the confidence score.

## Author
Claudio Pelleriti
