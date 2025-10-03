# K-Nearest Neighbors (KNN): Classification & Regression

This folder contains comprehensive implementations of **K-Nearest Neighbors (KNN)** algorithm for both classification and regression tasks. The notebooks provide intuitive explanations, practical implementations, and clear guidelines for when to use each approach.

## 📁 Project Structure

```
06-KNN/
├── 1-Implementation of KNN.ipynb        # KNN Classifier with Iris dataset
├── 2-When To Use KNN Classifier.ipynb  # Classification guidelines
├── 3-KNN Regressor.ipynb               # KNN for regression problems
├── 4-When_To_Use_KNN_Regressor.ipynb  # Regression guidelines
└── Conclusion.ipynb                    # Algorithm steps & distance metrics
```

## 🎯 Learning Objectives

- Master **KNN algorithm fundamentals** for both classification and regression
- Understand **distance metrics**: Euclidean vs Manhattan
- Implement **custom KNN** from scratch and with scikit-learn
- Learn **hyperparameter tuning** (choosing optimal K)
- Apply to **real-world datasets** and problems
- Understand **when to use KNN** vs other algorithms

## 🔧 Algorithm Overview

### **Core Concept: "Tell me who your neighbors are, and I'll tell you who you are"**

KNN is a **lazy learning** algorithm that makes predictions based on the similarity of data points:

1. **Choose K** (number of neighbors to consider)
2. **Calculate distances** to all training points
3. **Find K nearest neighbors**
4. **Make prediction**:
   - **Classification**: Majority vote of neighbor labels
   - **Regression**: Average of neighbor values

## 📊 KNN Classification

### **Implementation Highlights**
```python
# Custom KNN implementation concept
def knn_classify(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        # Calculate distances to all training points
        distances = [euclidean_distance(test_point, train_point) 
                    for train_point in X_train]
        
        # Find k nearest neighbors
        k_nearest = get_k_nearest(distances, y_train, k)
        
        # Majority vote
        prediction = majority_vote(k_nearest)
        predictions.append(prediction)
    
    return predictions
```

### **Iris Dataset Application**
- **Dataset**: 150 samples, 4 features, 3 classes
- **Features**: Sepal length/width, Petal length/width
- **Classes**: Setosa, Versicolor, Virginica
- **Performance**: High accuracy with proper K selection

### **Distance Metrics**

#### **Euclidean Distance (Default)**
```
d = √[(x₁-x₂)² + (y₁-y₂)²]
```
- **"As the crow flies"** - straight line distance
- **Best for**: Continuous features, geometric problems
- **Most common** choice for KNN

#### **Manhattan Distance**
```
d = |x₁-x₂| + |y₁-y₂|
```
- **"City block"** distance - grid-based movement
- **Best for**: High-dimensional data, categorical features
- **More robust** to outliers

## 📈 KNN Regression

### **Key Difference from Classification**
- **Classification**: Predicts categories (discrete labels)
- **Regression**: Predicts continuous numerical values

### **Prediction Method**
```python
# Instead of majority vote, use averaging
def knn_regress(neighbors_values, method='mean'):
    if method == 'mean':
        return np.mean(neighbors_values)
    elif method == 'weighted':
        # Weight by inverse distance
        return weighted_average(neighbors_values, distances)
```

### **Applications**
- **House price prediction** based on neighborhood
- **Stock price forecasting** using similar patterns
- **Temperature prediction** from nearby weather stations
- **Recommendation systems** with user similarity

## 🎯 When to Use KNN

### ✅ **Use KNN Classifier when:**

1. **Simple Classification Problems**
   - Clear category boundaries
   - "Similar things belong together" assumption holds
   - Need interpretable results

2. **Small to Medium Datasets**
   - KNN stores all training data
   - Computational cost increases with data size
   - Memory requirements can be high

3. **Non-parametric Problems**
   - No assumptions about data distribution
   - Complex decision boundaries
   - Local patterns more important than global trends

4. **Prototype Development**
   - Quick baseline model
   - No training phase required
   - Easy to implement and understand

### ✅ **Use KNN Regressor when:**

1. **Predicting Continuous Values**
   - House prices, temperatures, sales figures
   - Local similarity matters (nearby houses cost similar amounts)
   - Non-linear relationships present

2. **Smooth Prediction Surfaces**
   - Gradual changes in target values
   - Local averaging makes sense
   - Missing value imputation

### ❌ **Don't Use KNN when:**

1. **Large Datasets**
   - Millions of samples → very slow predictions
   - High memory requirements
   - Consider approximate methods or other algorithms

2. **High-Dimensional Data (Curse of Dimensionality)**
   - Many features make distance meaningless
   - All points become equidistant
   - Use dimensionality reduction first

3. **Imbalanced Classes**
   - Majority class dominates neighborhoods
   - Consider weighted KNN or other methods
   - Preprocessing may be needed

4. **Real-time Applications**
   - Prediction time increases with training set size
   - No model to store, must search every time
   - Consider pre-trained models instead

## 🔧 Hyperparameter Tuning

### **Choosing Optimal K**

#### **K Too Small (K=1)**
- **High variance**, low bias
- **Overfitting** to noise
- **Sensitive** to outliers
- **Jagged** decision boundaries

#### **K Too Large (K=N)**
- **High bias**, low variance
- **Underfitting** - always predicts majority class
- **Smooth** but potentially wrong boundaries
- **Loss of local patterns**

#### **Optimal K Selection**
```python
# Cross-validation approach
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
```

#### **Rules of Thumb**
- **Start with K = √N** (where N = number of samples)
- **Use odd numbers** to avoid ties in classification
- **Cross-validate** for optimal performance
- **Consider domain knowledge** about local vs global patterns

## 📊 Performance Characteristics

### **Advantages**
- **Simple to understand** and implement
- **No assumptions** about data distribution
- **Works with any distance metric**
- **Naturally handles multi-class** problems
- **Can capture complex patterns** locally
- **No training period** required

### **Disadvantages**
- **Computationally expensive** at prediction time
- **Memory intensive** (stores all training data)
- **Sensitive to irrelevant features**
- **Struggles with high dimensions**
- **Sensitive to data scaling**
- **Poor performance** with imbalanced data

## 🛠️ Implementation Best Practices

### **1. Data Preprocessing**
```python
from sklearn.preprocessing import StandardScaler

# Feature scaling is crucial for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### **2. Distance Metric Selection**
```python
from sklearn.neighbors import KNeighborsClassifier

# Try different distance metrics
knn_euclidean = KNeighborsClassifier(metric='euclidean')
knn_manhattan = KNeighborsClassifier(metric='manhattan')
knn_minkowski = KNeighborsClassifier(metric='minkowski', p=3)
```

### **3. Handling Categorical Features**
```python
# Use appropriate distance metrics for mixed data types
from sklearn.neighbors import KNeighborsClassifier

# Hamming distance for categorical features
knn = KNeighborsClassifier(metric='hamming')
```

### **4. Weighted KNN**
```python
# Weight neighbors by inverse distance
knn_weighted = KNeighborsClassifier(
    n_neighbors=5, 
    weights='distance'  # Closer neighbors have more influence
)
```

## 🚀 Getting Started

### **Prerequisites**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **Quick Start Guide**

1. **KNN Classification**
   ```bash
   jupyter notebook "1-Implementation of KNN.ipynb"
   ```

2. **Classification Guidelines**
   ```bash
   jupyter notebook "2-When To Use KNN Classifier.ipynb"
   ```

3. **KNN Regression**
   ```bash
   jupyter notebook "3-KNN Regressor.ipynb"
   ```

4. **Regression Guidelines**
   ```bash
   jupyter notebook "4-When_To_Use_KNN_Regressor.ipynb"
   ```

5. **Algorithm Deep Dive**
   ```bash
   jupyter notebook "Conclusion.ipynb"
   ```

## 🎯 Real-World Applications

### **Classification Examples**
- **Image Recognition**: Classify images based on pixel similarity
- **Recommendation Systems**: "Users like you also liked..."
- **Medical Diagnosis**: Classify symptoms based on similar cases
- **Fraud Detection**: Identify suspicious transactions
- **Text Classification**: Categorize documents by content similarity

### **Regression Examples**
- **Real Estate**: Predict house prices from neighborhood data
- **Finance**: Stock price prediction using similar market conditions
- **Weather**: Temperature forecasting from nearby stations
- **E-commerce**: Price optimization based on similar products
- **Healthcare**: Dosage prediction based on patient similarity

## 📈 Performance Optimization

### **1. Approximate Nearest Neighbors**
```python
# For large datasets, use approximate methods
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex  # Approximate library

# Build index for faster searches
index = AnnoyIndex(n_features, 'euclidean')
# Add items and build index
```

### **2. Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

# Reduce dimensions before applying KNN
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
```

### **3. Feature Selection**
```python
from sklearn.feature_selection import SelectKBest

# Select most relevant features
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)
```

## 🔍 Comparison with Other Algorithms

| Algorithm | Training Time | Prediction Time | Memory | Interpretability |
|-----------|---------------|-----------------|---------|------------------|
| **KNN** | O(1) | O(N) | High | High |
| **Decision Tree** | O(N log N) | O(log N) | Low | High |
| **SVM** | O(N²) | O(N) | Medium | Low |
| **Naive Bayes** | O(N) | O(1) | Low | Medium |
| **Random Forest** | O(N log N) | O(log N) | Medium | Medium |

## 📝 Key Takeaways

### **KNN is Perfect for:**
- **Prototype development** and baseline models
- **Small to medium datasets** with clear patterns
- **Problems where local similarity matters**
- **Non-parametric** classification/regression
- **Educational purposes** - easy to understand

### **Consider Alternatives when:**
- **Dataset is very large** (>100k samples)
- **High dimensionality** without feature selection
- **Real-time predictions** required
- **Imbalanced classes** without preprocessing
- **Memory constraints** are critical

---

*This comprehensive KNN implementation covers both theoretical foundations and practical applications, making it perfect for understanding one of machine learning's most intuitive algorithms while learning when and how to apply it effectively.*