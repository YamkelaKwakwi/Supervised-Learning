# Neural Network Regression Model

This project explores a neural network model developed to tackle a regression problem. We work with a dataset containing 1030 observations across 8 features and aim to predict a numeric target variable. 

## Project Overview

The project's primary objective is to evaluate the performance of different neural network configurations by tuning parameters such as learning rate and dropout rates across one, two, and three hidden-layer models. The model selection process was based on training Mean Squared Error (MSE) at different configurations, and the best model was chosen based on both performance and model simplicity.

### Table of Contents
1. [Introduction](#introduction)
2. [Data Exploration](#data-exploration)
3. [Model Building](#model-building)
4. [Training Evaluation](#training-evaluation)
5. [Test Performance](#test-performance)
6. [Conclusion](#conclusion)

## Introduction

We use a dataset with 1030 observations and 8 features to predict a target variable that has a range between 2.33 and 82.60 (mean: 35.82). Due to the lack of information on the dataset’s features, no feature engineering was applied, and all features were scaled to prepare for training. No missing values were present in the dataset.

## Data Exploration

The dataset was partitioned into an 80/20 split, allocating 824 observations for training and 206 for testing. Scaling was performed to ensure consistent performance in model training.

## Model Building

### Training Configurations

We trained models with three configurations:
- **One Hidden Layer**
- **Two Hidden Layers**
- **Three Hidden Layers**

Each configuration was trained using varying combinations of:
- **Learning Rates** (LR): 0.0001, 0.001, 0.01
- **Dropout Rates** (DO): 0.2, 0.3, 0.5

To visualize model performance, we generated line plots showing how the MSE changed over training epochs across different learning rates and dropout rates. The MSE values were log-transformed to aid interpretability.

### Observations on Model Performance

#### One Hidden Layer
- **Best configuration**: LR = 0.01, DO = 0.2
- **Worst configuration**: LR = 0.0001, DO = 0.5

#### Two Hidden Layers
- Similar performance trend with **LR = 0.01, DO = 0.2** performing the best.

#### Three Hidden Layers
- Models with **LR = 0.0001** performed poorly due to stalled learning.
- Models with **LR = 0.01** converged efficiently, showing the best performance.

## Training Evaluation

After training models with all configurations, we identified that epoch 50 was an optimal point, where the MSE across configurations stabilized. We compared the models' MSE at this epoch, and the top-performing models had the following configurations:
- Learning Rates of 0.01 or 0.001
- Dropout Rates of 0.2 or 0.3

### Model Comparison

A bar graph was used to visualize the training MSE across the 27 model configurations at epoch 50, with the best-performing models exhibiting the lowest MSE.

## Test Performance

From the 27 configurations, the best model was selected based on its balance between performance and simplicity. This was:
- **Model Configuration**: One hidden layer, LR = 0.01, DO = 0.2
- **Test MSE**: 44.67

This model performed effectively on the test set, achieving a low test MSE, indicating good generalization.

## Conclusion

Through extensive evaluation and tuning of model hyperparameters, we selected a simple, effective neural network model for regression. The chosen configuration demonstrated strong performance in predicting unseen data, with low test MSE and minimal overfitting.

## Plots and Figures

Figures included in this project:
1. **Training MSE Plots** for one, two, and three hidden-layer configurations.
2. **Bar Graph of Training MSE at Epoch 50** for all configurations.
3. **Table of Predicted vs. Actual Values** for a subset of test observations.

## Future Improvements

Potential enhancements to this project could include:
- Experimenting with additional feature engineering techniques.
- Implementing regularization methods to further control overfitting.
- Exploring advanced architectures, such as convolutional layers, to identify patterns in the features.

## Getting Started

1. Clone the repository.
2. Install all required packages.
3. Run the Markdown file to replicate the model training and evaluation process.

---

**Author**: Yamkela Kwakwi  
