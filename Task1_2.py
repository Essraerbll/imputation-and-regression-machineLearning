import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

# Step 1: Generate a Toy Regression Dataset
# ---------------------------------------------------
# Create a dataset with 5 features and 3 dependent target variables. 
# Use random values for the features and a linear relationship for the targets.
np.random.seed(42)  # Seed for reproducibility
data = pd.DataFrame({
    'feature_1': np.random.uniform(10, 50, 5000),
    'feature_2': np.random.uniform(100, 200, 5000),
    'feature_3': np.random.uniform(0, 1, 5000),
    'feature_4': np.random.uniform(50, 100, 5000),
    'feature_5': np.random.uniform(-100, -50, 5000),
})

# Targets are generated using linear combinations of the features, with added noise.
data['target_1'] = 100 * data['feature_1'] + 50 * data['feature_2'] + np.random.normal(0, 5, 5000)
data['target_2'] = 5 * data['feature_3'] - 2 * data['feature_4'] + np.random.normal(0, 15, 5000)
data['target_3'] = 9 * data['feature_5'] + 0.5 * data['feature_1'] + np.random.normal(0, 10, 5000)

# Save a copy of the original dataset for comparison.
data_original = data.copy()

# Step 2: Introduce Missing Values
# ---------------------------------------------------
# Randomly select 2% of the dataset rows and set their 'feature_1' values to NaN.
missing_indices = np.random.choice(data.index, int(0.02 * len(data)), replace=False)
data.loc[missing_indices, 'feature_1'] = np.nan

# Save the original dataset with missing values for future reference.
data_original.to_csv("original_data.csv", index=False)

# Step 3: Random Imputation
# ---------------------------------------------------
# Replace missing values in 'feature_1' with random values drawn from its original range.
min_val = data['feature_1'].min()
max_val = data['feature_1'].max()
random_imputed_values = np.random.uniform(min_val, max_val, size=missing_indices.shape[0])
data.loc[missing_indices, 'feature_1'] = random_imputed_values

# Save the dataset after random imputation.
data.to_csv("random_imp.csv", index=False)

# Step 4: Regression Imputation
# ---------------------------------------------------
# Prepare training data for regression imputation by excluding rows with missing values.
X_train = data.dropna()[['feature_2', 'feature_3', 'feature_4', 'feature_5', 'target_1', 'target_2', 'target_3']]
y_train = data.dropna()['feature_1']

# Extract the rows with missing values for prediction.
X_missing = data.loc[missing_indices, ['feature_2', 'feature_3', 'feature_4', 'feature_5', 'target_1', 'target_2', 'target_3']]

# Scale the features to the range [0, 1] using MinMaxScaler.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_missing_scaled = scaler.transform(X_missing)

# Use Ridge regression with hyperparameter tuning to predict missing values.
param_grid = {'alpha': np.logspace(-5, 5, 11)}  # Wide range of alpha values for Ridge
ridge_model = Ridge()
grid_search_ridge = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train_scaled, y_train)
best_ridge = grid_search_ridge.best_estimator_  # Best Ridge model from GridSearch

# Predict missing values for 'feature_1' using the trained Ridge model.
predictions_ridge = best_ridge.predict(X_missing_scaled)
data.loc[missing_indices, 'feature_1'] = predictions_ridge

# Save the dataset after regression imputation.
data.to_csv("regression_imp.csv", index=False)

# Step 5: Scatter Plot and MSE Calculation
# ---------------------------------------------------
# Define a helper function to plot actual vs predicted values and compute MSE.
def plot_relationship(ax, actual_values, estimated_values, target_actual, target_name):
    ax.scatter(actual_values, target_actual, color='blue', label='Actual Values', alpha=0.9, marker='*')
    ax.scatter(estimated_values, target_actual, color='red', label='Predicted Values', alpha=0.9, marker='*')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel(target_name)
    ax.set_title(f'Feature 1 vs {target_name} (Actual vs Predicted)')
    ax.legend()
    mse = mean_squared_error(actual_values, estimated_values)
    return mse

# Plot for each target variable to compare actual vs predicted imputed values.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
mse_1 = plot_relationship(axes[0], data_original.loc[missing_indices, 'feature_1'], predictions_ridge, 
                          data.loc[missing_indices, 'target_1'], 'Target 1')
mse_2 = plot_relationship(axes[1], data_original.loc[missing_indices, 'feature_1'], predictions_ridge, 
                          data.loc[missing_indices, 'target_2'], 'Target 2')
mse_3 = plot_relationship(axes[2], data_original.loc[missing_indices, 'feature_1'], predictions_ridge, 
                          data.loc[missing_indices, 'target_3'], 'Target 3')
plt.tight_layout()
plt.show()

# Print the MSE for each scatter plot.
print(f"MSE for Feature 1 vs Target 1: {mse_1}")
print(f"MSE for Feature 1 vs Target 2: {mse_2}")
print(f"MSE for Feature 1 vs Target 3: {mse_3}")

# Step 6: Train-Test Split for Models
# ---------------------------------------------------
# Prepare datasets for random and regression imputation.
X_random = data.drop(columns='feature_1')
y_random = data['feature_1']

X_original = data_original.drop(columns='feature_1')
y_original = data_original['feature_1']

# Split each dataset into 70% training and 30% testing sets.
X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X_random, y_random, test_size=0.3, random_state=42)
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.3, random_state=42)

# Step 7: Neural Network Models
# ---------------------------------------------------
# Define a neural network model pipeline for both imputations.
pipeline_random = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(200, 50), activation='relu', solver='adam', max_iter=5000,
                         alpha=0.0001, learning_rate='adaptive', random_state=42, early_stopping=True))
])

pipeline_original = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(200, 50), activation='relu', solver='adam', max_iter=5000,
                         alpha=0.0001, learning_rate='adaptive', random_state=42, early_stopping=True))
])

# Train models.
pipeline_random.fit(X_train_random, y_train_random)
pipeline_original.fit(X_train_original, y_train_original)

# Predict and evaluate MSE.
mse_train_random = mean_squared_error(y_train_random, pipeline_random.predict(X_train_random))
mse_test_random = mean_squared_error(y_test_random, pipeline_random.predict(X_test_random))

mse_train_original = mean_squared_error(y_train_original, pipeline_original.predict(X_train_original))
mse_test_original = mean_squared_error(y_test_original, pipeline_original.predict(X_test_original))

# Create a DataFrame to compare the results.
mse_comparison = pd.DataFrame({
    'Imputation Type': ['Original Data', 'Random Imputation'],
    'MSE (Train)': [mse_train_original, mse_train_random],
    'MSE (Test)': [mse_test_original, mse_test_random]
})

print(mse_comparison)
