# Autonomous Vehicle Perception Module

Machine-learning project for recognizing German traffic signs as part of an autonomous-vehicle perception system.

## Goal

Classify 43 German traffic-sign categories from the German Traffic Sign Recognition Benchmark (GTSRB) dataset and compare classical machine-learning, deep-learning, transfer-learning, recurrent, and transformer-based approaches.

## Methods

- Image resizing and normalization
- HOG feature extraction
- Standard scaling and PCA dimensionality reduction
- K-Nearest Neighbors and Gaussian Naive Bayes
- Hard and soft voting classifiers
- Bagging classifiers
- Convolutional Neural Networks
- CNN autoencoder for image reconstruction and denoising
- GRU, Simple RNN, and LSTM models
- MobileNetV2 transfer learning
- Vision Transformer fine-tuning

## Dataset

The notebook downloads the GTSRB dataset from Kaggle using `kagglehub`:

<https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>

The dataset includes training images arranged by class, test images, labels, and metadata for all 43 traffic-sign categories.

## Workflow

1. Download and load the dataset.
2. Split the training data into training and validation sets.
3. Preprocess images and extract HOG features.
4. Train and evaluate classical classifiers with and without PCA.
5. Train CNN, recurrent, autoencoder, MobileNetV2, and Vision Transformer models.
6. Compare models using accuracy, loss, reconstruction MSE, and confusion matrices.
7. Save trained models, predictions, features, and training histories.

## Project Structure

```text
.
├── Project.ipynb
├── README.md
├── features/
│   ├── x_train.npy
│   ├── x_val.npy
│   ├── x_test.npy
│   ├── x_train_scaled.npy
│   ├── x_val_scaled.npy
│   ├── x_test_scaled.npy
│   ├── x_train_noisy.npy
│   └── x_train_reconstructed.npy
├── models/
│   ├── autoencoder.h5
│   └── encoder.h5
└── mobilenet_results/
	├── mobilenet_traffic.keras
	├── history.json
	├── y_pred_mobilenet.npy
	├── y_pred_probs_mobilenet.npy
	└── y_true_mobilenet.npy
```

## Requirements

```bash
pip install numpy pandas matplotlib seaborn pillow scikit-learn scikit-image tensorflow torch transformers tqdm kagglehub joblib
```

## Running the Project

Open `Project.ipynb` in Jupyter Notebook or Visual Studio Code and run the cells in order. Internet access is needed to download the Kaggle dataset, pretrained MobileNetV2 weights, and the pretrained Vision Transformer.

The deep-learning and Vision Transformer sections may require substantial memory and a GPU for practical training times.

## Results

The notebook generates:

- Validation and test accuracy measurements
- Training and validation loss curves
- Model comparison plots
- Confusion matrices
- Clean, noisy, and reconstructed image comparisons
- Saved trained models and prediction arrays
- Saved feature arrays and autoencoder training history

Exact metric values are produced when the notebook is executed.

## Source Label

Training / Practice project using the German Traffic Sign Recognition Benchmark dataset.