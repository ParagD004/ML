# Decision Trees - Module 07

A comprehensive guide to Decision Tree algorithms for both classification and regression tasks, featuring practical implementations and clear usage guidelines.

## 📚 Contents

### 1. Decision Tree Classifier (`1-Decision Tree.ipynb`)
- **Dataset**: Iris flower classification (150 samples, 4 features, 3 classes)
- **Implementation**: Basic DecisionTreeClassifier with scikit-learn
- **Features**: Model training, prediction, accuracy evaluation, and tree visualization
- **Output**: Clear decision rules for species classification

### 2. When to Use Decision Tree Classifier (`2-When_To_Use_DT-Classifier.ipynb`)
- **Guidelines**: Clear criteria for choosing decision tree classifiers
- **Use Cases**:
  - Simple, interpretable decision rules needed
  - Understanding "why" behind predictions is important
  - Categorical or simple numerical data
  - Smaller datasets (doesn't require millions of examples)
  - Quick, good-enough models with fast training

### 3. Decision Tree Regressor - Diabetes Prediction (`3-Decision_Tree_Regressor_Diabetes_Prediction.ipynb`)
- **Dataset**: Diabetes progression prediction
- **Implementation**: DecisionTreeRegressor for continuous value prediction
- **Features**: Data preprocessing, model training, performance metrics, and visualization
- **Application**: Predicting numerical health outcomes

### 4. When to Use Decision Tree Regressor (`4-When_To_Use_DT-Regressor.ipynb`)
- **Purpose**: Predicting continuous numbers (prices, temperatures, scores)
- **Use Cases**:
  - Simple rules explaining numerical predictions
  - Easy-to-interpret flowchart-like models
  - Mixed categorical and numerical data
  - Fast, transparent predictions over extreme accuracy

### 5. Final Touch (`final_touch.ipynb`)
- **Core Concepts**:
  - Decision trees work for both classification and regression
  - Supervised learning with labeled training data
  - Tree structure: nodes (splits), leaves (predictions)
  - Pure vs impure splits explanation
- **Key Principles**: "If-then" rule learning from data patterns

## 🎯 Key Concepts

- **Classification**: Predicts class labels (categories)
- **Regression**: Predicts continuous numerical values
- **Nodes**: Decision points with feature-based conditions
- **Leaves**: Final predictions (class labels or numbers)
- **Splitting**: Dividing data based on feature values for maximum purity

## 🛠️ Libraries Used

- **scikit-learn**: DecisionTreeClassifier, DecisionTreeRegressor
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Visualization

## ✅ Advantages

- **Interpretable**: Clear "if-then" rules
- **No preprocessing**: Handles categorical and numerical data
- **Fast training**: Quick model development
- **Feature importance**: Identifies key decision factors
- **Non-parametric**: No assumptions about data distribution

## 🚀 Getting Started

1. Start with `1-Decision Tree.ipynb` for basic classification
2. Review `2-When_To_Use_DT-Classifier.ipynb` for usage guidelines
3. Explore `3-Decision_Tree_Regressor_Diabetes_Prediction.ipynb` for regression
4. Check `4-When_To_Use_DT-Regressor.ipynb` for regression guidelines
5. Conclude with `final_touch.ipynb` for comprehensive understanding

## 🎯 Learning Outcomes

- Implement decision trees for classification and regression
- Understand when to choose decision trees over other algorithms
- Interpret tree-based decision rules
- Apply decision trees to real-world problems
- Evaluate model performance and visualize results

---
