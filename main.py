from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import pyswarms as ps
import math

from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("data_20.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
y = (y != 0).astype(int)

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)

df_res = pd.DataFrame(X_res, columns=X.columns)  
df_res['target'] = y_res
num_class = len(np.unique(y_res))

desired_size_per_class = 1000

data = pd.concat([
    df_class.sample(n=desired_size_per_class, random_state=42)
    for _, df_class in df_res.groupby('target')
])

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Initialize patterns and correlations for training data
pattern_list = []
train_correlations = np.zeros(len(X_train))

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

# Optimized objective function with subsampling
def objective_for_optimization(params, existing_pattern_list, existing_correlations, sample_indices):
    corr_threshold = params[0]
    pattern_index_float = params[1]
    # Scale weight from 0-1 to -10 to 10 range
    pattern_weight = params[2] * 20 - 10  # Scale from 0-1 to -10 to 10
    
    # Convert index to integer and bound it
    max_index = len(existing_pattern_list)
    pattern_index = min(max_index, math.floor(pattern_index_float * (max_index + 1)))
    
    # Extract pattern parameters
    new_pattern = params[3:]
    
    # Use only the sampled indices
    sampled_X = X_train_standardized[sample_indices]
    sampled_y = y_train.iloc[sample_indices]
    sampled_correlations = existing_correlations[sample_indices]
    
    # Make a copy of correlations (avoid modifying original)
    temp_correlations = sampled_correlations.copy()
    
    # Standardize pattern (once per evaluation)
    pattern_mean = np.mean(new_pattern)
    pattern_std_dev = np.std(new_pattern) or 1
    pattern_std = (new_pattern - pattern_mean) / pattern_std_dev
    
    # Calculate new pattern correlation efficiently for sampled data
    pattern_correlations = np.sum(sampled_X * pattern_std, axis=1) / len(pattern_std)
    # Apply ReLU - set negative correlations to zero
    pattern_correlations = np.maximum(0, pattern_correlations)
    # Apply weight to pattern correlations
    pattern_correlations = pattern_correlations * pattern_weight
    
    # Handle existing pattern replacement
    if pattern_index < len(existing_pattern_list):
        old_pattern, old_weight = existing_pattern_list[pattern_index]
        old_pattern_mean = np.mean(old_pattern)
        old_pattern_std_dev = np.std(old_pattern) or 1
        old_pattern_std = (old_pattern - old_pattern_mean) / old_pattern_std_dev
        old_correlations = np.sum(sampled_X * old_pattern_std, axis=1) / len(old_pattern_std)
        # Apply ReLU to old correlations
        old_correlations = np.maximum(0, old_correlations)
        # Apply weight to old correlations
        old_correlations = old_correlations * old_weight
        temp_correlations -= old_correlations
    
    # Add new pattern correlation
    temp_correlations += pattern_correlations
    
    # Fast evaluation
    preds = (temp_correlations > corr_threshold).astype(int)
    pos_count = np.sum(preds)
    if pos_count == 0 or pos_count == len(preds):
        return -1
    
    # Only calculate AUC if predictions are valid
    return roc_auc_score(sampled_y, temp_correlations)

# Simplified objective for pyswarms (avoiding unnecessary return values)
def pyswarms_objective(particles):
    n_particles = particles.shape[0]
    j = np.zeros(n_particles)
    
    # Sample 50% of training data for stochastic approximation
    sample_size = len(X_train) // 2
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    
    for i in range(n_particles):
        auc = objective_for_optimization(particles[i], pattern_list, train_correlations, sample_indices)
        j[i] = -auc
    
    return j

# Faster correlation calculation for a dataset
def calculate_correlations(X_standardized, patterns):
    correlations = np.zeros(len(X_standardized))
    
    for pattern, weight in patterns:
        pattern_mean = np.mean(pattern)
        pattern_std_dev = np.std(pattern) or 1
        pattern_std = (pattern - pattern_mean) / pattern_std_dev
        pattern_correlations = np.sum(X_standardized * pattern_std, axis=1) / len(pattern_std)
        # Apply ReLU - set negative correlations to zero
        pattern_correlations = np.maximum(0, pattern_correlations)
        # Apply weight to correlations
        pattern_correlations = pattern_correlations * weight
        correlations += pattern_correlations
    
    return correlations

# Optimization parameters - expanded for weight
n_particles = 300
max_iterations = 100

# PSO options - tuned for faster convergence
options = {
    'c1': 0.7,  # Higher cognitive parameter for faster convergence 
    'c2': 0.5,  # Social parameter
    'w': 0.6,   # Lower inertia for faster convergence
    'k': 2,     # Fewer neighbors to check
    'p': 2      # Minkowski p-norm
}

# Configure bounds - added dimension for weight (now -10 to 10)
def create_bounds():
    # One dimension for threshold, one for pattern index, one for weight (-10 to 10), and 20 for pattern
    max_bound = np.ones(23)  # All parameters normalized to 0-1 range
    min_bound = np.zeros(23)
    return (min_bound, max_bound)

# Run optimization until AUC stops improving
previous_train_auc = 0
current_train_auc = 0
iteration = 0
best_test_auc = 0
best_threshold = 0

while True:
    # Configure bounds with weight parameter
    bounds = create_bounds()
    
    # Initialize optimizer with additional dimension for weight
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=23,  # Now 23 dimensions (added weight)
        options=options,
        bounds=bounds
    )
    
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(
        pyswarms_objective, 
        iters=max_iterations,
        verbose=False
    )
    
    # Extract parameters
    corr_threshold = best_pos[0]
    pattern_index_float = best_pos[1]
    # Scale weight from 0-1 to -10 to 10 range
    pattern_weight = best_pos[2] * 20 - 10  # Scale from 0-1 to -10 to 10
    new_pattern = best_pos[3:]
    
    # Convert index to integer
    max_index = len(pattern_list)
    pattern_index = min(max_index, math.floor(pattern_index_float * (max_index + 1)))
    
    # Action type
    action = "Updating" if pattern_index < len(pattern_list) else "Creating new"
    
    # Standardize pattern
    pattern_mean = np.mean(new_pattern)
    pattern_std_dev = np.std(new_pattern) or 1
    pattern_std = (new_pattern - pattern_mean) / pattern_std_dev
    
    # Calculate correlation on full training set
    pattern_correlations = np.sum(X_train_standardized * pattern_std, axis=1) / len(pattern_std)
    # Apply ReLU - set negative correlations to zero
    pattern_correlations = np.maximum(0, pattern_correlations)
    # Apply weight to correlations
    pattern_correlations = pattern_correlations * pattern_weight
    
    # Handle pattern update/creation
    if pattern_index < len(pattern_list):
        old_pattern, old_weight = pattern_list[pattern_index]
        old_pattern_mean = np.mean(old_pattern)
        old_pattern_std_dev = np.std(old_pattern) or 1
        old_pattern_std = (old_pattern - old_pattern_mean) / old_pattern_std_dev
        old_correlations = np.sum(X_train_standardized * old_pattern_std, axis=1) / len(old_pattern_std)
        # Apply ReLU to old correlations
        old_correlations = np.maximum(0, old_correlations)
        # Apply weight to old correlations
        old_correlations = old_correlations * old_weight
        train_correlations -= old_correlations
        pattern_list[pattern_index] = (new_pattern, pattern_weight)
    else:
        pattern_list.append((new_pattern, pattern_weight))
    
    # Update correlations
    train_correlations += pattern_correlations
    
    # Calculate test performance
    test_correlations = calculate_correlations(X_test_standardized, pattern_list)
    
    # Evaluate performance on full datasets
    previous_train_auc = current_train_auc
    current_train_auc = roc_auc_score(y_train, train_correlations)
    test_auc = roc_auc_score(y_test, test_correlations)
    
    # Track best performance
    if test_auc > best_test_auc:
        best_test_auc = test_auc
        best_threshold = corr_threshold
    
    print(f"Iteration {iteration+1}, {action} pattern at index {pattern_index}, Weight: {pattern_weight:.2f}, Train AUC: {current_train_auc:.4f}, Test AUC: {test_auc:.4f}")
    
    # Increment iteration counter
    iteration += 1
    
    # Check if AUC decreased (break condition)
    if iteration > 1 and current_train_auc < previous_train_auc:
        print(f"Training AUC decreased from {previous_train_auc:.4f} to {current_train_auc:.4f}. Stopping.")
        break

# Final evaluation (using full test set)
final_test_correlations = calculate_correlations(X_test_standardized, pattern_list)
final_test_preds = (final_test_correlations > best_threshold).astype(int)
final_test_accuracy = accuracy_score(y_test, final_test_preds)
final_test_auc = roc_auc_score(y_test, final_test_correlations)

print("\n--- Final Evaluation ---")
print(f"Test Accuracy: {final_test_accuracy:.4f}")
print(f"Test AUC: {final_test_auc:.4f}")
print(f"Number of patterns: {len(pattern_list)}")
print(f"Best threshold: {best_threshold:.4f}")