# Gradient Boosting

A comprehensive guide to Gradient Boosting algorithms for both classification and regression tasks using scikit-learn.

## Overview

Gradient Boosting is an ensemble learning method that builds models sequentially, where each new model corrects the errors made by previous models. It combines weak learners (typically decision trees) to create a strong predictive model.

### Key Concepts

- **Sequential Learning**: Models are built one after another, each learning from previous mistakes
- **Gradient Descent**: Uses gradient descent to minimize loss function
- **Weak Learners**: Typically uses shallow decision trees as base learners
- **Additive Model**: Final prediction is the sum of all weak learner predictions

## Directory Structure

```
10-Gradient Boosting/
├── Classification/
│   ├── 1-GBC implementation.ipynb          # Basic GBC implementation
│   ├── 2-About-Gradient-Boosting-classifier.ipynb  # Theory and concepts
│   └── auc.png                             # AUC visualization
├── Regression/
│   ├── 1-GBR implementation.ipynb          # Basic GBR implementation
│   └── 2-About-gradient-boosting-regression.ipynb  # Theory and concepts
└── README.md                               # This file
```

## Classification (GBC)

### Gradient Boosting Classifier Features

- **Multi-class Support**: Handles multiple classes naturally
- **Feature Importance**: Provides built-in feature importance scores
- **Probability Estimates**: Can output class probabilities
- **Robust to Outliers**: Less sensitive to outliers compared to other algorithms

### Key Parameters

```python
GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinks contribution of each tree
    max_depth=3,             # Maximum depth of trees
    min_samples_split=2,     # Minimum samples to split node
    min_samples_leaf=1,      # Minimum samples in leaf node
    max_features=None,       # Number of features for best split
    random_state=None        # Random seed for reproducibility
)
```

### Use Cases

- **Image Classification**: Digit recognition, object classification
- **Text Classification**: Sentiment analysis, spam detection
- **Medical Diagnosis**: Disease prediction, risk assessment
- **Financial Modeling**: Credit scoring, fraud detection

## Regression (GBR)

### Gradient Boosting Regressor Features

- **Continuous Predictions**: Predicts continuous target values
- **Non-linear Relationships**: Captures complex patterns in data
- **Feature Selection**: Automatic feature importance ranking
- **Regularization**: Built-in regularization through shrinkage

### Key Parameters

```python
GradientBoostingRegressor(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinks contribution of each tree
    max_depth=3,             # Maximum depth of trees
    min_samples_split=2,     # Minimum samples to split node
    min_samples_leaf=1,      # Minimum samples in leaf node
    loss='squared_error',    # Loss function to optimize
    alpha=0.9               # Quantile for huber/quantile loss
)
```

### Use Cases

- **House Price Prediction**: Real estate valuation
- **Stock Price Forecasting**: Financial time series
- **Sales Forecasting**: Business analytics
- **Scientific Modeling**: Environmental and biological data

## Advantages

✅ **High Accuracy**: Often achieves excellent predictive performance
✅ **Feature Importance**: Provides interpretable feature rankings
✅ **Handles Mixed Data**: Works with numerical and categorical features
✅ **No Data Preprocessing**: Minimal need for feature scaling
✅ **Robust**: Less prone to overfitting with proper tuning
✅ **Versatile**: Works for both classification and regression

## Disadvantages

❌ **Training Time**: Can be slow to train with many estimators
❌ **Memory Usage**: Requires more memory than single models
❌ **Hyperparameter Tuning**: Many parameters to optimize
❌ **Sequential Nature**: Cannot be easily parallelized
❌ **Overfitting Risk**: Can overfit with too many estimators

## Performance Optimization

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
```

### Early Stopping

```python
# Monitor validation score to prevent overfitting
gbc = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=5,
    tol=1e-4
)
```

## Best Practices

1. **Start Simple**: Begin with default parameters, then tune
2. **Cross-Validation**: Always use CV for hyperparameter tuning
3. **Feature Engineering**: Good features improve performance significantly
4. **Early Stopping**: Use validation sets to prevent overfitting
5. **Learning Rate**: Lower learning rates often work better with more estimators
6. **Tree Depth**: Keep trees shallow (3-8 levels) for better generalization

## Getting Started

1. **Classification**: Start with `Classification/1-GBC implementation.ipynb`
2. **Regression**: Start with `Regression/1-GBR implementation.ipynb`
3. **Theory**: Read the "About" notebooks for deeper understanding
4. **Experimentation**: Try different datasets and parameters

## Common Issues and Solutions

### Overfitting
- Reduce `learning_rate`
- Increase `min_samples_split` and `min_samples_leaf`
- Use early stopping
- Reduce `max_depth`

### Underfitting
- Increase `n_estimators`
- Increase `learning_rate`
- Increase `max_depth`
- Reduce regularization parameters

### Slow Training
- Reduce `n_estimators`
- Increase `learning_rate`
- Use `max_features` to subsample features
- Consider using `subsample` for stochastic gradient boosting



---

*This directory provides hands-on experience with gradient boosting algorithms, from basic implementations to advanced techniques and visualizations.*