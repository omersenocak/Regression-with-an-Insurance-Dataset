# Premium Amount Prediction with LightGBM

## Overview
This project focuses on predicting the **Premium Amount** for customers using a dataset from Kaggle. The primary objective is to develop an optimized regression model using various machine learning algorithms and evaluate their performance using the **Root Mean Squared Logarithmic Error (RMSLE)** metric.

The project culminates with predictions on a test dataset, leveraging the best-performing model (LightGBM) with advanced hyperparameter tuning.

---

## Steps Followed

### 1. Data Preparation
- Handled missing values for numerical and categorical features.
- Extracted date features from `Policy Start Date`.
- Performed one-hot encoding for categorical variables.

### 2. Model Selection
- Evaluated 10 machine learning models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - XGBoost
  - LightGBM
- The models were evaluated based on the RMSLE metric, and LightGBM was identified as the best performer.

### 3. Hyperparameter Tuning
- Performed advanced hyperparameter optimization for LightGBM using **Optuna**.
- Conducted Bayesian Optimization to fine-tune parameters, achieving improved performance.

### 4. Feature Importance Analysis
- Visualized feature importances for the LightGBM model to understand the key predictors of `Premium Amount`.

### 5. Predictions on Test Data
- Applied the same preprocessing steps to the test dataset.
- Generated predictions for `Premium Amount` using the optimized LightGBM model.

---

## Dataset Information

### Train Dataset
- **Entries**: 1,200,000
- **Features**: 21 columns
- **Target**: `Premium Amount`

### Test Dataset
- **Entries**: 800,000
- **Features**: 20 columns
- Predictions for the `Premium Amount` are saved in the output file.

---

## Usage

### Prerequisites
- Python 3.8+
- Required Libraries:
  - pandas
  - numpy
  - scikit-learn
  - lightgbm
  - optuna
  - matplotlib

Install the dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Project
1. **Preprocessing and Model Training**
   - Run the notebook/script to preprocess the data and train the models.
   - The best model (LightGBM) is saved after hyperparameter tuning.

2. **Testing and Predictions**
   - Preprocess the test dataset using the provided pipeline.
   - Generate predictions for `Premium Amount` using the optimized LightGBM model.
   - Save the predictions to `test_predictions.csv`.

---

## File Structure
```plaintext
├── data/
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset
├── notebooks/
│   ├── model_training.ipynb  # Model training and evaluation
│   ├── test_predictions.ipynb # Test data processing and predictions
├── src/
│   ├── preprocess.py       # Preprocessing functions
│   ├── train.py            # Model training scripts
│   ├── predict.py          # Test prediction scripts
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## Results
- **Best Model**: LightGBM
- **Final RMSLE on Validation Data**: `1.1329`

---

## Feature Importances
A bar plot of the top 10 most important features influencing the `Premium Amount` prediction:

![Feature Importances](images/feature_importances.png)

---

## License
This project is released under the MIT License. See `LICENSE` for details.

---

## Contributing
Contributions are welcome! Feel free to submit a pull request or raise an issue for discussions.

---

## Citation
Walter Reade and Elizabeth Park. Regression with an Insurance Dataset. https://kaggle.com/competitions/playground-series-s4e12, 2024. Kaggle.

