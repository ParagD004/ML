# Support Vector Machines (SVM) - Module 04

This module provides a comprehensive implementation and explanation of Support Vector Machines, covering both classification (SVC) and regression (SVR) techniques with practical examples and kernel methods.

## 📚 Contents

### 1. Basic SVC Implementation (`1-Basic_SVC_Implementation.ipynb`)
- **Objective**: Introduction to Support Vector Classification with synthetic data
- **Key Topics**:
  - Creating synthetic classification datasets
  - Basic SVC implementation using scikit-learn
  - Data visualization and scatter plots
  - Model training and evaluation
  - Grid search for hyperparameter tuning
  - Performance metrics and accuracy assessment

### 2. SVM Kernels Implementation (`2-SVM_Kernels_Implementation.ipynb`)
- **Objective**: Deep dive into different SVM kernel functions and their applications
- **Key Topics**:
  - Kernel trick explanation and intuition
  - Linear, RBF, Polynomial, and Sigmoid kernels
  - Kernel parameter tuning
  - Visualization of decision boundaries
  - Interactive plotting with Plotly
  - Comparative analysis of different kernels
  - Performance evaluation across kernel types

### 3. Support Vector Regression Implementation (`3-Support_Vector_Regression_Implementation.ipynb`)
- **Objective**: Implementation of SVR for continuous value prediction
- **Key Topics**:
  - SVR vs SVC differences
  - Epsilon-insensitive loss function
  - Hyperparameter tuning for SVR
  - Model evaluation metrics for regression
  - Visualization of regression results
  - Comparison with other regression techniques

### 4. When to Use SVM (`4-When_To_Use_SVM.ipynb`)
- **Objective**: Guidelines and best practices for SVM usage
- **Key Topics**:
  - **SVC (Support Vector Classification)**:
    - Task type: Classification → output is a class label
    - Goal: Find optimal hyperplane to separate classes
    - Loss function: Hinge loss for margin maximization
    - Use case: When target variable is categorical
  - **SVR (Support Vector Regression)**:
    - Task type: Regression → output is continuous value
    - Goal: Fit function while ignoring small errors (within tolerance ε)
    - Loss function: Epsilon-insensitive loss
    - Use case: When target variable is numeric/continuous

### 5. Example Walk-Through (`5-Example-Walk-Through.ipynb`)
- **Objective**: Practical example demonstrating SVM concepts
- **Key Topics**:
  - Visual explanation with SVM diagram
  - 2D classification problem (points in/out of circle)
  - Step-by-step walkthrough of SVM decision-making
  - Kernel transformation visualization
  - Real-world problem solving approach

### 6. Final Touch (`final_touch.ipynb`)
- **Objective**: Summary and key terminology consolidation
- **Key Topics**:
  - **Loss Functions Comparison**:
    - **SVC Hinge Loss**: "Be strict, separate groups clearly"
    - **SVR Epsilon Loss**: "Small errors are okay, big errors are not"
  - Key terminologies and concepts
  - Best practices summary
  - Implementation guidelines

## 🛠️ Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Static plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and tools
- **plotly**: Interactive visualizations
- **IPython**: Enhanced interactive Python

## 🎯 Learning Objectives

By completing this module, you will:

1. **Understand SVM Fundamentals**:
   - Difference between SVC and SVR
   - Concept of support vectors and hyperplanes
   - Margin maximization principle

2. **Master Kernel Methods**:
   - Linear, RBF, Polynomial, and Sigmoid kernels
   - When and how to use different kernels
   - Kernel parameter optimization

3. **Implement Practical Solutions**:
   - Classification problems with SVC
   - Regression problems with SVR
   - Hyperparameter tuning techniques
   - Model evaluation and validation

4. **Apply Best Practices**:
   - When to choose SVM over other algorithms
   - How to handle different data types and structures
   - Performance optimization techniques

## 🚀 Getting Started

1. Start with `1-Basic_SVC_Implementation.ipynb` for foundational concepts
2. Progress through `2-SVM_Kernels_Implementation.ipynb` for advanced kernel methods
3. Explore `3-Support_Vector_Regression_Implementation.ipynb` for regression applications
4. Review `4-When_To_Use_SVM.ipynb` for decision-making guidelines
5. Work through `5-Example-Walk-Through.ipynb` for practical understanding
6. Conclude with `final_touch.ipynb` for concept consolidation

## 📈 Key Concepts Covered

- **Support Vectors**: Critical data points that define the decision boundary
- **Hyperplane**: Decision boundary that separates classes
- **Margin**: Distance between hyperplane and nearest support vectors
- **Kernel Trick**: Mapping data to higher dimensions for linear separation
- **C Parameter**: Regularization parameter controlling overfitting
- **Gamma Parameter**: Influence of individual training examples
- **Epsilon Parameter**: Tolerance for SVR predictions

## 🎨 Visualizations

The module includes various visualization techniques:
- Decision boundary plots
- Support vector highlighting
- Kernel transformation demonstrations
- Performance comparison charts
- Interactive Plotly visualizations

---
