from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

from visualizations import visualize_iteration_results
from patternextraction import PatternOptimizer

# Load and preprocess data
data = pd.read_csv("data_20.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = (y != 0).astype(int)

# Undersample to balance classes
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)

df_res = pd.DataFrame(X_res, columns=X.columns)  
df_res['target'] = y_res
num_class = len(np.unique(y_res))

# Sample equally from both classes
desired_size_per_class = 1000

data = pd.concat([
    df_class.sample(n=desired_size_per_class, random_state=42)
    for _, df_class in df_res.groupby('target')
])

X = data.drop('target', axis=1)
y = data['target']

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Convert data to numpy arrays once (avoid repeated conversion)
X_train_array = X_train.values
X_test_array = X_test.values

# Pre-compute standardized training data
X_train_means = X_train_array.mean(axis=1, keepdims=True)
X_train_stds = X_train_array.std(axis=1, keepdims=True)
X_train_stds[X_train_stds == 0] = 1
X_train_standardized = (X_train_array - X_train_means) / X_train_stds

# Pre-compute standardized test data
X_test_means = X_test_array.mean(axis=1, keepdims=True)
X_test_stds = X_test_array.std(axis=1, keepdims=True)
X_test_stds[X_test_stds == 0] = 1
X_test_standardized = (X_test_array - X_test_means) / X_test_stds

# Create pattern optimizer
optimizer = PatternOptimizer(X_train_standardized, y_train, X_test_standardized, y_test)

# Run optimization with visualization
best_model, best_patterns, best_starts, best_ends, best_activations, best_control_points = optimizer.run_optimization(
    max_iterations=10,
    visualize_func=lambda iteration, patterns, starts, ends, activations, model, X_train_std, **kwargs: 
        visualize_iteration_results(
            iteration, patterns, starts, ends, activations, model, 
            X_train_std, y_train=y_train, **kwargs
        )
)

# The results are already printed by the PatternOptimizer class