# Breast Cancer Diagnosis Using Machine Learning

## Overview

This project implements and evaluates multiple machine learning algorithms for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The objective is to classify tumors as **Malignant** or **Benign** based on features extracted from Fine Needle Aspiration (FNA) biopsy images.

## Dataset

* Source: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset
* Total Samples: 569
* Features: 30
* Classes:

  * Benign: 357
  * Malignant: 212
* Missing Values: None

## Models Implemented

1. LightGBM
2. Multi-Layer Perceptron (MLP)
3. Gradient Boosting Classifier
4. Gaussian Process Classifier

## Preprocessing

* Feature Scaling using StandardScaler
* Train-Test Split (80:20)
* Model Training and Evaluation
* Cross Validation

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* AUC-ROC
* Confusion Matrix
* Cross Validation Score

## Results

| Model             | Accuracy | AUC-ROC |
| ----------------- | -------- | ------- |
| LightGBM          | 95.61%   | 0.9888  |
| MLP               | 94.74%   | 0.9944  |
| Gradient Boosting | 95.61%   | 0.9907  |
| Gaussian Process  | 97.37%   | 0.9934  |

### Best Performing Model

* Gaussian Process Classifier
* Accuracy: 97.37%

### Highest AUC-ROC

* MLP Neural Network
* AUC-ROC: 0.9944

## Technologies Used

* Python
* Scikit-Learn
* LightGBM
* NumPy
* Pandas
* Matplotlib
* Seaborn

## Project Structure

```text
breast-cancer-diagnosis-ml/
│
├── data/
├── images/
├── presentation/
├── src/
├── requirements.txt
└── README.md
```

## Reference

Nemade, V. & Fegade, V. (2023). Machine Learning Techniques for Breast Cancer Prediction. Procedia Computer Science, 218, 1314–1320.

