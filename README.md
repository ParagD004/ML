# Machine Learning Algorithms Collection

A comprehensive collection of machine learning algorithms implemented in Python using scikit-learn and other popular libraries. This repository provides hands-on examples, theoretical explanations, and practical implementations for both beginners and advanced practitioners.

## 📚 Repository Structure

```
Machine-Learning-Algorithms/
├── 01-Linear Regression/           # Simple and multiple linear regression
├── 02-Ridge and Lasso/            # Regularized linear regression
├── 03-Logistic Regression/        # Classification using logistic regression
├── 04-SVM/                        # Support Vector Machines
├── 05-Naive Baye's/               # Probabilistic classification
├── 06-KNN/                        # K-Nearest Neighbors
├── 07-Decision Tree/              # Tree-based learning
├── 08-Random Forest/              # Ensemble of decision trees
├── 09-Adaboost/                   # Adaptive boosting
├── 10-Gradient Boosting/          # Gradient boosting machines
├── 11-XGBoost/                    # Extreme gradient boosting
└── README.md                      # This file
```

## 🎯 Learning Path

### Beginner Track
1. **Linear Regression** → Understanding the basics of supervised learning
2. **Logistic Regression** → Introduction to classification
3. **Decision Trees** → Interpretable non-linear models
4. **K-Nearest Neighbors** → Instance-based learning

### Intermediate Track
5. **Ridge and Lasso** → Regularization techniques
6. **Naive Bayes** → Probabilistic approaches
7. **Support Vector Machines** → Margin-based classification
8. **Random Forest** → Ensemble methods introduction

### Advanced Track
9. **AdaBoost** → Adaptive boosting algorithms
10. **Gradient Boosting** → Sequential ensemble learning
11. **XGBoost** → State-of-the-art gradient boosting

## 📊 Algorithm Comparison

| Algorithm | Type | Interpretability | Speed | Accuracy | Overfitting Risk |
|-----------|------|------------------|-------|----------|------------------|
| Linear Regression | Regression | High | Very Fast | Medium | Low |
| Ridge/Lasso | Regression | High | Fast | Medium | Low |
| Logistic Regression | Classification | High | Fast | Medium | Low |
| SVM | Both | Low | Medium | High | Medium |
| Naive Bayes | Classification | Medium | Very Fast | Medium | Low |
| KNN | Both | Medium | Slow | Medium | High |
| Decision Tree | Both | High | Fast | Medium | High |
| Random Forest | Both | Medium | Medium | High | Low |
| AdaBoost | Both | Low | Medium | High | Medium |
| Gradient Boosting | Both | Low | Slow | Very High | Medium |
| XGBoost | Both | Low | Fast | Very High | Low |

## 🔧 Use Case Guide

### Regression Problems
- **House Price Prediction**: Linear Regression, Ridge/Lasso, Random Forest, XGBoost
- **Stock Price Forecasting**: SVM, Gradient Boosting, XGBoost
- **Sales Forecasting**: Linear Regression, Random Forest, XGBoost
- **Energy Consumption**: Decision Trees, Random Forest, Gradient Boosting

### Classification Problems
- **Email Spam Detection**: Naive Bayes, Logistic Regression, SVM
- **Image Recognition**: SVM, Random Forest, XGBoost
- **Medical Diagnosis**: Decision Trees, Random Forest, XGBoost
- **Customer Churn**: Logistic Regression, Random Forest, XGBoost
- **Fraud Detection**: SVM, Random Forest, XGBoost

### Specific Scenarios
- **Small Dataset**: Logistic Regression, SVM, Naive Bayes
- **Large Dataset**: XGBoost, Random Forest, Linear Regression
- **High Interpretability**: Decision Trees, Linear/Logistic Regression
- **High Accuracy**: XGBoost, Gradient Boosting, Random Forest
- **Fast Prediction**: Naive Bayes, KNN, Linear Models
- **Categorical Features**: Decision Trees, Random Forest, XGBoost

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost jupyter
```

### Basic Workflow
```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 2. Load and prepare data
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Choose and train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

## 📈 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 🛠️ Common Preprocessing Steps

### Data Cleaning
```python
# Handle missing values
data.fillna(data.mean(), inplace=True)  # Numerical
data.fillna(data.mode().iloc[0], inplace=True)  # Categorical

# Remove duplicates
data.drop_duplicates(inplace=True)
```

### Feature Engineering
```python
# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])

# Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Feature Selection
```python
# Correlation-based selection
correlation_matrix = data.corr()
high_corr_features = correlation_matrix[abs(correlation_matrix) > 0.8]

# Statistical selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

## 🎯 Model Selection Guidelines

### Choose Based on Data Size
- **Small (< 1K samples)**: Logistic Regression, SVM, Naive Bayes
- **Medium (1K - 100K)**: Random Forest, SVM, XGBoost
- **Large (> 100K)**: XGBoost, Linear Models, Random Forest

### Choose Based on Feature Count
- **Few features (< 10)**: Any algorithm
- **Many features (10-100)**: Random Forest, XGBoost, Regularized models
- **High-dimensional (> 100)**: Regularized models, SVM, Random Forest

### Choose Based on Requirements
- **Speed Priority**: Naive Bayes, Linear Models, KNN
- **Accuracy Priority**: XGBoost, Gradient Boosting, Random Forest
- **Interpretability Priority**: Decision Trees, Linear Models
- **Robustness Priority**: Random Forest, XGBoost

## 📚 Learning Resources

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani

### Online Courses
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- edX MIT Introduction to Machine Learning
- Udacity Machine Learning Engineer Nanodegree

### Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

## 🔍 Advanced Topics

### Ensemble Methods
- **Voting Classifiers**: Combine multiple algorithms
- **Stacking**: Use meta-learner to combine predictions
- **Blending**: Weighted combination of models

### Hyperparameter Tuning
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameters
- **Bayesian Optimization**: Smart parameter search
- **Optuna**: Advanced hyperparameter optimization

### Cross-Validation Strategies
- **K-Fold**: Standard cross-validation
- **Stratified K-Fold**: Maintains class distribution
- **Time Series Split**: For temporal data
- **Leave-One-Out**: For small datasets

## 🚨 Common Pitfalls

### Data Leakage
- Using future information to predict past events
- Including target variable in features
- Data preprocessing before train/test split

### Overfitting
- Too complex models for small datasets
- Not using validation sets
- Ignoring regularization

### Underfitting
- Too simple models for complex data
- Insufficient feature engineering
- Poor hyperparameter tuning

### Evaluation Issues
- Using wrong metrics for the problem
- Not using cross-validation
- Ignoring class imbalance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your implementation with proper documentation
4. Include examples and visualizations
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn community for excellent documentation
- Kaggle for datasets and competitions
- Open source contributors for various libraries
- Academic researchers for algorithm development

---

*This repository serves as a comprehensive guide to machine learning algorithms, providing both theoretical understanding and practical implementation skills.*