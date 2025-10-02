# Advanced Machine Learning Models for Forest Fire Prediction

This folder contains advanced machine learning implementations for predicting forest fire weather index (FWI) using the Algerian Forest Fires dataset. The project focuses on regularization techniques and model optimization. I have also integrated wandb in this folder but it is completely optional, you can ignore the wandb part and study the main part without it.

## 📁 Project Structure

```
02/
├── 1-Ridge and Lasso.ipynb          # Ridge and Lasso regression implementation
├── 2-Advanced_Model_Training.ipynb   # Advanced model training techniques
├── 3-final_touch.ipynb              # Final model refinements
├── 4-Cv_and_tuning.ipynb           # Cross-validation and hyperparameter tuning
├── Algerian_forest_fires_cleaned.csv # Cleaned dataset
├── lasso_model.pkl                  # Trained Lasso model
├── ridge_model.pkl                  # Trained Ridge model
├── standard_scaler.pkl              # Fitted StandardScaler
└── wandb/                           # Weights & Biases experiment tracking
```

## 🎯 Project Objectives

- Implement regularization techniques (Ridge and Lasso regression)
- Perform advanced model training with feature selection
- Apply cross-validation and hyperparameter tuning
- Track experiments using Weights & Biases
- Build production-ready models for forest fire prediction

## 📊 Dataset Features

The Algerian Forest Fires dataset includes meteorological and fire weather indices:

- **Temperature**: Air temperature (°C)
- **RH**: Relative humidity (%)
- **Ws**: Wind speed (km/h)
- **Rain**: Rainfall (mm)
- **FFMC**: Fine Fuel Moisture Code
- **DMC**: Duff Moisture Code
- **DC**: Drought Code
- **ISI**: Initial Spread Index
- **BUI**: Buildup Index
- **Classes**: Fire occurrence (binary: fire/not fire)
- **Region**: Geographic region (0 or 1)

**Target Variable**: **FWI** (Fire Weather Index)

## 🔧 Key Techniques Implemented

### 1. Regularization Methods
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Lasso Regression**: L1 regularization with feature selection capabilities
- **ElasticNet**: Combination of L1 and L2 regularization

### 2. Model Training & Evaluation
- Train-test split (75/25)
- Feature scaling using StandardScaler
- Cross-validation for robust model evaluation
- Hyperparameter tuning using GridSearchCV

### 3. Experiment Tracking
- Weights & Biases integration for experiment monitoring
- Model versioning and performance tracking
- Automated logging of metrics and parameters

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn wandb joblib
```

### Running the Notebooks

1. **Ridge and Lasso Implementation**
   ```bash
   jupyter notebook "1-Ridge and Lasso.ipynb"
   ```

2. **Advanced Model Training**
   ```bash
   jupyter notebook "2-Advanced_Model_Training.ipynb"
   ```

3. **Final Model Refinements**
   ```bash
   jupyter notebook "3-final_touch.ipynb"
   ```

4. **Cross-Validation and Tuning**
   ```bash
   jupyter notebook "4-Cv_and_tuning.ipynb"
   ```

## 📈 Model Performance

The models are evaluated using:
- **Mean Squared Error (MSE)**
- **R² Score**
- **Cross-validation scores**
- **Feature importance analysis**

## 🔄 Model Persistence

Trained models are saved as pickle files:
- `ridge_model.pkl`: Optimized Ridge regression model
- `lasso_model.pkl`: Optimized Lasso regression model
- `standard_scaler.pkl`: Fitted feature scaler

### Loading Saved Models
```python
import joblib

# Load models
ridge_model = joblib.load('ridge_model.pkl')
lasso_model = joblib.load('lasso_model.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Make predictions
predictions = ridge_model.predict(scaler.transform(new_data))
```

## 📊 Experiment Tracking

This project uses Weights & Biases for:
- Real-time metric tracking
- Hyperparameter optimization
- Model comparison
- Experiment reproducibility

### 🔗 W&B Project Dashboard
View live experiments and results: [Ridge & Lasso Regression Project](https://wandb.ai/paragd004-dr-d-y-patil-vidyapeeth/ridge-lasso-regression)

## 🎯 Key Insights

- **Feature Correlation**: Strong correlations between fire weather indices
- **Regularization Benefits**: Improved generalization through L1/L2 penalties
- **Feature Selection**: Lasso regression identifies most important predictors
- **Model Stability**: Cross-validation ensures robust performance estimates

## 🔗 Related Files

- See `../01/` folder for basic regression implementations
- Dataset source: Algerian Forest Fires Dataset
- Original research: Forest fire prediction using meteorological data

## 📝 Notes

- All models use standardized features for optimal performance
- Hyperparameters are tuned using cross-validation
- Model artifacts are version-controlled for reproducibility
- Experiment logs are available in the `wandb/` directory

---

*This project demonstrates advanced machine learning techniques for environmental prediction tasks, focusing on regularization methods and robust model evaluation practices.*