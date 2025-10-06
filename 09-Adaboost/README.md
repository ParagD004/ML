# AdaBoost: Adaptive Boosting Algorithm

This folder contains implementations of **AdaBoost (Adaptive Boosting)** for both classification and regression tasks. AdaBoost combines multiple weak learners to create a strong ensemble model.

## 📁 Project Structure

```
09-Adaboost/
├── Classifier/
│   ├── 1-AdaBoost Classifier.ipynb           # Implementation with scikit-learn
│   ├── 2-About-Ada-Boost-Classification.ipynb # Step-by-step explanation
│   └── Iris.csv                              # Dataset for classification
└── Regressor/
    ├── 1-AdaBoost Regressor.ipynb            # Regression implementation
    └── 2-About-Ada-Boost-Regression.ipynb    # Regression explanation
```

## 🔧 Algorithm Overview

**AdaBoost** sequentially builds weak learners, focusing on previously misclassified examples:

1. **Initialize equal weights** for all training samples
2. **Train weak learner** on weighted data
3. **Calculate error** and learner importance
4. **Update weights**: Increase for misclassified samples
5. **Repeat** until desired number of learners
6. **Combine predictions** using weighted voting/averaging

## 📊 Key Implementations

### **AdaBoost Classifier**
- **Iris Dataset**: Multi-class flower classification
- **Weak Learners**: Decision stumps (shallow decision trees)
- **Voting**: Weighted majority vote for final prediction
- **Error Handling**: Focuses on misclassified examples

### **AdaBoost Regressor**
- **House Price Prediction**: Continuous value estimation
- **Weak Learners**: Simple regression models
- **Averaging**: Weighted average of predictions
- **Error Reduction**: Sequential improvement of predictions

## 🎯 Key Concepts

### **Adaptive Learning**
- **Weight Updates**: Misclassified samples get higher weights
- **Sequential Training**: Each learner learns from previous mistakes
- **Error Minimization**: Focuses computational effort on difficult cases

### **Ensemble Strength**
- **Weak to Strong**: Combines many simple models
- **Bias Reduction**: Improves overall prediction accuracy
- **Overfitting Control**: Generally resistant to overfitting

## 🚀 Getting Started

### **Prerequisites**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Notebooks Overview**
1. **Classification**: Iris dataset with decision stumps
2. **Classification Theory**: Step-by-step algorithm explanation
3. **Regression**: House price prediction example
4. **Regression Theory**: Mathematical foundation and examples

## 📈 Performance Characteristics

### **Advantages**
- Converts weak learners into strong classifier
- Automatically handles feature selection
- Generally resistant to overfitting
- Works well with simple base learners

### **Disadvantages**
- Sensitive to noise and outliers
- Can be slow with large datasets
- Performance depends on weak learner choice
- May overfit with very noisy data

---

*This AdaBoost implementation demonstrates both theoretical foundations and practical applications of adaptive boosting for classification and regression tasks.*