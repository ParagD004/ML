# K-Nearest Neighbors (KNN): Classification & Regression

This folder contains implementations of **K-Nearest Neighbors (KNN)** algorithm for both classification and regression tasks with performance optimization techniques.

## 📁 Project Structure

```
06-KNN/
├── 1-Implementation of KNN.ipynb        # KNN Classifier with Iris dataset
├── 2-When To Use KNN Classifier.ipynb  # Classification guidelines
├── 3-KNN Regressor.ipynb               # KNN for regression problems
├── 4-When_To_Use_KNN_Regressor.ipynb  # Regression guidelines
├── 5-Caching nearest neighbour.ipynb   # Performance optimization with caching
└── Conclusion.ipynb                    # Algorithm steps & distance metrics
```

## 🔧 Algorithm Overview

KNN is a **lazy learning** algorithm that makes predictions based on similarity:

1. **Choose K** (number of neighbors)
2. **Calculate distances** to all training points
3. **Find K nearest neighbors**
4. **Make prediction**:
   - **Classification**: Majority vote of neighbor labels
   - **Regression**: Average of neighbor values

## 📊 Key Implementations

### **KNN Classification**
- **Iris Dataset**: 150 samples, 4 features, 3 classes
- **Custom Implementation**: From-scratch distance calculations
- **Scikit-learn Integration**: Production-ready implementation

### **KNN Regression**
- **Continuous Value Prediction**: Average neighbor values
- **Weighted Predictions**: Distance-based weighting
- **Applications**: House prices, stock prediction, weather forecasting

### **Caching Optimization**
- **Performance Enhancement**: Store computed distances
- **Memory vs Speed Trade-off**: Faster repeated predictions
- **Large Dataset Handling**: Efficient neighbor searches

## 🔍 Distance Metrics

### **Euclidean Distance**
```
d = √[(x₁-x₂)² + (y₁-y₂)²]
```
- Straight-line distance ("as the crow flies")
- Best for continuous features

### **Manhattan Distance**
```
d = |x₁-x₂| + |y₁-y₂|
```
- Grid-based distance ("city blocks")
- More robust to outliers

## 🚀 Getting Started

### **Prerequisites**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Notebooks Overview**
1. **Basic KNN**: Classification with Iris dataset
2. **Usage Guidelines**: When to apply KNN classifier
3. **Regression**: Predicting continuous values
4. **Regression Guidelines**: When to use KNN regressor
5. **Caching**: Performance optimization techniques
6. **Conclusion**: Algorithm details and distance metrics

## 📈 Performance Characteristics

### **Advantages**
- Simple to understand and implement
- No training period required
- Works with any distance metric
- Handles multi-class problems naturally

### **Disadvantages**
- Computationally expensive at prediction time
- Memory intensive (stores all training data)
- Sensitive to irrelevant features and data scaling
- Struggles with high-dimensional data

## 🛠️ Optimization Techniques

- **Feature Scaling**: StandardScaler for consistent distances
- **Caching**: Store computed distances for repeated queries
- **Approximate Methods**: For large datasets
- **Dimensionality Reduction**: PCA before applying KNN

---

*This KNN implementation covers fundamental concepts, practical applications, and performance optimization techniques for both classification and regression tasks.*