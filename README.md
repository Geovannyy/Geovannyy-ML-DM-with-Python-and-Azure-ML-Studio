# Machine Learning and Data Mining Project

## Project Overview
This repository contains the implementation of three machine learning tasks analysing different datasets:

1. **Classification Analysis (Task 1)**
   - Binary classification on bank marketing dataset
   - Implemented in both Python and Azure ML
   - Includes SMOTE and SMOTEENN handling for imbalanced data

2. **Clustering Analysis (Task 2)**
   - Customer segmentation on online shoppers intention dataset
   - Feature engineering for temporal and engagement metrics
   - PCA-based dimensionality reduction

3. **Sentiment Analysis (Task 3)**
   - Recipe reviews sentiment analysis
   - LSTM-based deep learning implementation
   - Traditional ML baselines (SVM, Logistic Regression, Naïve Bayes)

## Repository Structure
```
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and pre-processed data
│   ├── interim/               # Intermediate data (NOT USED)
│   └── external/              # External data sources (NOT USED)
├── models/                    # Trained models
│   ├── Task_1/
│   ├── Task_2/
│   └── Task_3/
├── notebooks/                 # Jupyter notebooks
│   ├── Task_1_Classification.ipynb
│   ├── Task_2_Clustering.ipynb
│   └── Task_3_Sentiment_Analysis.ipynb
├── reports/
│   └── figures/              # Generated visualizations
└── references/               # Data dictionaries, manuals, etc. (NOT USED)
```

## Datasets
- Task 1: Bank Marketing Dataset (`data/raw/bank.csv`)
- Task 2: Online Shoppers Intention Dataset (`data/raw/online_shoppers_intention.csv`)
- Task 3: Recipe Reviews Dataset (`data/raw/Recipe Reviews and User Feedback Dataset.csv`)

## Requirements
- Python 3.12.3
- CUDA-compatible GPU (for TensorFlow)
- Azure Machine Learning Studio account (for Task 1b)

## Key Dependencies
- TensorFlow 2.18.0
- scikit-learn 1.5.2
- pandas 2.2.3
- numpy 2.0.2
- matplotlib 3.9.2
- seaborn 0.13.2
- nltk 3.9.1
- wordcloud 1.9.4
- imbalanced-learn 0.12.4

## Installation

1. Clone the repository:

git clone https://github.com/Geovannyy/Geovannyy-ML-DM-with-Python-and-Azure-ML-Studio.git


2. Create WSL2 environment from the provided environment.yml:

python12.3 - m venv -f environment.yml

3. Activate the environment:

source /{project}/bin/activate Machine-Learning-and-Data-Mining


4. Verify installation:

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


### Note on Deep Learning Support
This project uses TensorFlow 2.18.0 with CUDA support. The environment includes:
- NVIDIA CUDA Runtime 12.5
- cuDNN 9.3.0
- NVIDIA CUDA Toolkit dependencies

If you don't have a CUDA-compatible GPU, the deep learning components will default to CPU run.

## Usage

### Task 1: Classification Analysis

jupyter notebook ../notebooks/Task_1_Classification.ipynb

- Implements binary classification for bank marketing prediction
- Features both Python and Azure ML implementations
- Includes comprehensive EDA and model comparison

### Task 2: Clustering Analysis

jupyter notebook ../notebooks/Task_2_Clustering.ipynb

- Customer segmentation analysis
- Feature engineering and PCA implementation
- Comparative analysis of clustering algorithms

### Task 3: Sentiment Analysis

jupyter notebook ../notebooks/Task_3_Sentiment_Analysis.ipynb

- Text preprocessing and sentiment classification
- LSTM model implementation
- Comparative analysis with traditional ML approaches

## Model Results

### Task 1: Classification
- Location: `models/Task_1/evaluation_results.json`
- Preprocessed data: `data/processed/Task_1_processed/`
- Visualizations: `reports/figures/Task_1/`

### Task 2: Clustering
- Preprocessed data: `data/processed/Task_2_processed/`
- Visualizations: `reports/figures/Task_2/`

### Task 3: Sentiment Analysis
- Best model: `models/Task_3/lstm_best.keras`
- Preprocessed data: `data/processed/Task_3_processed/`
- Visualizations: `reports/figures/Task_3/`

## Project Structure Details

### Data Organization
- `data/raw/`: Original datasets
- `data/processed/`: Cleaned and transformed datasets
- `data/interim/`: Intermediate processing stages
- `data/external/`: External reference data

### Code Organization
- `notebooks/`: Main analysis notebooks
- `models/`: Saved model artifacts
- `reports/figures/`: Generated visualizations and plots

## Author Information
- Name: [Your Name]
- Student ID: [Your ID]
- Module: Machine Learning and Data Mining
- University of Salford
- Academic Year: 2024-25

## Acknowledgments
- Datasets sourced from public repositories