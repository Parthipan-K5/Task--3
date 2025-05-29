# House Price Prediction Project

## Project Overview
This project implements a machine learning solution to predict house prices based on various features. It uses multiple regression models to provide accurate price predictions and compares their performance.

## Dataset Description
The dataset contains the following features:
- `price`: Target variable - the price of the house
- `area`: Area of the house in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `stories`: Number of stories
- `mainroad`: Whether the house is connected to the main road (yes/no)
- `guestroom`: Whether the house has a guest room (yes/no)
- `basement`: Whether the house has a basement (yes/no)
- `hotwaterheating`: Whether the house has hot water heating (yes/no)
- `airconditioning`: Whether the house has air conditioning (yes/no)
- `parking`: Number of parking spots
- `prefarea`: Whether the house is in a preferred area (yes/no)
- `furnishingstatus`: Furnishing status (furnished, semi-furnished, unfurnished)

## Implementation Details

### Data Preprocessing
```python
# Convert binary columns
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                 'airconditioning', 'prefarea']
df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# Skip one-hot encoding if furnishingstatus is already encoded
if 'furnishingstatus' in df.columns:
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Feature scaling
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
numeric_transformer = StandardScaler()
```

### Model Pipeline
The project implements the following models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regression
- K-Nearest Neighbors

### Model Performance
Based on the evaluation metrics:

1. Linear Regression & Lasso Regression:
   - R² Score: 0.5464
   - RMSE: 1,514,173.55
   - Best performing models overall

2. Ridge Regression:
   - R² Score: 0.5462
   - RMSE: 1,514,551.85
   - Very close performance to Linear/Lasso

3. Gradient Boosting:
   - R² Score: 0.5258
   - RMSE: 1,548,211.90
   - Good balance of performance and complexity

## Setup Instructions

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Data preparation:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

3. Run the preprocessing pipeline:
```python
# Load data
df = pd.read_csv('Housing.csv')

# Preprocess data
# (Use preprocessing code from above)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Best Practices

1. Data Preprocessing:
   - Always check for missing values
   - Handle categorical variables appropriately
   - Scale numeric features
   - Use cross-validation for model evaluation

2. Model Selection:
   - Start with simple models (Linear Regression)
   - Progress to more complex models if needed
   - Use hyperparameter tuning for optimization
   - Consider model interpretability vs performance

3. Evaluation:
   - Use multiple metrics (RMSE, MAE, R²)
   - Compare models systematically
   - Consider computational costs
   - Validate on holdout test set

## Troubleshooting

### Common Issues

1. Categorical Variable Encoding:
   ```python
   # Check if furnishingstatus is already encoded
   encoded_columns = df.columns[df.columns.str.startswith('furnishingstatus_')]
   if len(encoded_columns) == 0:
       df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
   ```

2. Feature Scaling:
   ```python
   # Ensure proper scaling of numeric features
   scaler = StandardScaler()
   df[numeric_features] = scaler.fit_transform(df[numeric_features])
   ```

3. Model Performance:
   ```python
   # Cross-validation for more robust evaluation
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X, y, cv=5, scoring='r2')
   print(f"Mean R² score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
   ```

## Future Improvements

1. Feature Engineering:
   - Create interaction terms
   - Add polynomial features
   - Calculate price per square foot
   - Generate location-based features

2. Advanced Modeling:
   - Implement ensemble methods
   - Try deep learning approaches
   - Use stacking/blending techniques
   - Implement time-series analysis for price trends

3. Deployment:
   - Create API endpoint
   - Build web interface
   - Add real-time predictions
   - Implement model monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.