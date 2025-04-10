import numpy as np
import math
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.special import comb

# Function to generate Bezier curve points
def bernstein_poly(i, n, t):
    """Bernstein polynomial basis function"""
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, num_points=20):
    """
    Generate a Bezier curve from control points
    
    Parameters:
    -----------
    control_points : array-like
        Control points for the Bezier curve (x, y coordinates)
    num_points : int, optional
        Number of points to generate on the curve
        
    Returns:
    --------
    array : Bezier curve points
    """
    n = len(control_points) - 1  # Degree of the curve
    t = np.linspace(0, 1, num_points)
    curve = np.zeros(num_points)
    
    for i, point in enumerate(control_points):
        curve += point * np.array([bernstein_poly(i, n, t_) for t_ in t])
    
    return curve

def extract_control_points(pattern, start, end):
    """
    Extract approximate control points for a Bezier curve from a pattern
    within the active region
    
    Parameters:
    -----------
    pattern : array-like
        The pattern to extract control points from
    start : int
        Start index of the active region
    end : int
        End index of the active region
        
    Returns:
    --------
    array : Control points in the full-array format
    """
    # Initialize control points array
    control_points = [0.0] * len(pattern)
    
    # Calculate positions within active region
    active_region_size = end - start + 1
    control_indices = [
        start,  # First control point at start
        start + active_region_size // 3,  # Second at 1/3
        start + 2 * active_region_size // 3,  # Third at 2/3
        end  # Last at end
    ]
    
    # Set control point values
    for i, idx in enumerate(control_indices):
        control_points[idx] = pattern[idx]
    
    return control_points

# Faster correlation calculation for a dataset that returns per-pattern correlations
def calculate_correlations_as_features(X_standardized, patterns, starts=None, ends=None, activation='relu'):
    """Calculate correlations for each pattern and return as feature matrix
    
    Parameters:
    -----------
    X_standardized : array-like
        Standardized input data
    patterns : list of arrays
        List of patterns to calculate correlations against
    starts : list of int, optional
        Start indices for each pattern
    ends : list of int, optional
        End indices for each pattern
    activation : str or list, optional
        Activation function to apply to correlations.
        If string, same activation for all patterns.
        If list, should contain one activation function name per pattern.
        Options: 'linear', 'relu', 'tanh', 'sigmoid', 'quadratic', 'softplus'
        Default: 'relu'
    """
    feature_matrix = np.zeros((len(X_standardized), len(patterns)))
    
    # Handle activation as string or list
    if isinstance(activation, str):
        # Same activation for all patterns
        act_list = [activation] * len(patterns)
    else:
        # Different activation for each pattern
        act_list = activation
    
    for i, pattern in enumerate(patterns):
        start = starts[i] if starts is not None and i < len(starts) else 0
        end = ends[i] if ends is not None and i < len(ends) else len(pattern) - 1
        
        pattern_mean = np.mean(pattern[start:end+1])
        pattern_std_dev = np.std(pattern[start:end+1]) or 1
        
        pattern_std = pattern.copy()
        pattern_std[start:end+1] = (pattern[start:end+1] - pattern_mean) / pattern_std_dev
        
        # Zero out pattern values outside the start-end range
        pattern_mask = np.zeros_like(pattern_std)
        pattern_mask[start:end+1] = 1
        pattern_std = pattern_std * pattern_mask
        
        pattern_correlations = np.sum(X_standardized * pattern_std, axis=1) / (end - start + 1)
        
        # Get current pattern's activation
        current_activation = act_list[i]
        
        # Apply activation function
        if current_activation == 'relu':
            # ReLU - set negative correlations to zero
            pattern_correlations = np.maximum(0, pattern_correlations)
        elif current_activation == 'linear':
            # Linear - keep as is
            pass
        elif current_activation == 'tanh':
            # Tanh - hyperbolic tangent
            pattern_correlations = np.tanh(pattern_correlations)
        elif current_activation == 'sigmoid':
            # Sigmoid - logistic function
            pattern_correlations = 1 / (1 + np.exp(-pattern_correlations))
        elif current_activation == 'quadratic':
            # Quadratic - square the values
            pattern_correlations = pattern_correlations ** 2
        elif current_activation == 'softplus':
            # Softplus - smooth ReLU: log(1 + exp(x))
            pattern_correlations = np.log1p(np.exp(pattern_correlations))
        else:
            raise ValueError(f"Unknown activation function: {current_activation}")
            
        feature_matrix[:, i] = pattern_correlations
    
    return feature_matrix

# Default LightGBM parameters
def get_lgb_params():
    return {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'min_child_samples': 40, 
        'random_state': 42,
        'verbosity': -1,
    }

# Function to train and evaluate a model with given patterns
def train_and_evaluate(X_train_standardized, y_train, X_test_standardized, y_test, patterns, starts, ends, activations='relu'):
    """
    Train and evaluate a model with given patterns
    
    Parameters:
    -----------
    X_train_standardized : array-like
        Standardized training data
    y_train : array-like
        Training labels
    X_test_standardized : array-like
        Standardized test data
    y_test : array-like
        Test labels
    patterns : list of arrays
        List of patterns to use for feature extraction
    starts : list of int
        Start indices for each pattern
    ends : list of int
        End indices for each pattern
    activations : str or list, optional
        Activation functions to apply. If string, same activation is used for all patterns.
        If list, should contain one activation function name per pattern.
        Default: 'relu'
    
    Returns:
    --------
    model : trained model
    train_auc : float
        AUC score on training data
    test_auc : float
        AUC score on test data
    val_auc : float
        AUC score on validation data
    """
    # Handle activations as a single string or list
    train_features = calculate_correlations_as_features(
        X_train_standardized, patterns, starts, ends, activations
    )
    test_features = calculate_correlations_as_features(
        X_test_standardized, patterns, starts, ends, activations
    )
    
    # Create validation split for evaluation
    X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
        train_features, y_train, test_size=0.2, random_state=42
    )
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(**get_lgb_params())
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    
    # Train model
    model.fit(
        X_train_lgb, 
        y_train_lgb, 
        eval_set=[(X_val_lgb, y_val_lgb)],
        callbacks=callbacks,
        eval_metric='auc'
    )
    
    # Get predictions
    train_preds = model.predict_proba(train_features)[:, 1]
    val_preds = model.predict_proba(X_val_lgb)[:, 1]
    test_preds = model.predict_proba(test_features)[:, 1]
    
    # Calculate AUC scores
    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val_lgb, val_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    return model, train_auc, test_auc, val_auc

# Optimized objective function for Optuna
def objective_function(trial, sample_indices, X_train_standardized, y_train, 
                       current_patterns, current_starts, current_ends):
    """Objective function for Optuna optimization"""
    # Sample pattern parameters
    pattern_index_float = trial.suggest_float('pattern_index', 0.0, 1.0)
    pattern_start = trial.suggest_int('pattern_start', 0, 14)  # Ensure room for at least 5 positions
    
    # Ensure pattern is at least 5 positions long
    min_end = pattern_start + 4  # +4 to make sure there are at least 5 positions (start to end inclusive)
    pattern_end = trial.suggest_int('pattern_end', min_end, 19)
    
    # Convert index to integer and bound it
    max_index = len(current_patterns)
    pattern_index = min(max_index, math.floor(pattern_index_float * (max_index + 1)))
    
    # Use Bezier control points only within the active region
    # Generate 4 control points within the active region
    control_points = [0.0] * 20  # Initialize with zeros
    # Place control points only within the active region
    active_region_size = pattern_end - pattern_start + 1
    control_indices = [
        pattern_start,  # First control point at start
        pattern_start + active_region_size // 3,  # Second at 1/3
        pattern_start + 2 * active_region_size // 3,  # Third at 2/3
        pattern_end  # Last at end
    ]
    
    # Generate values for control points (between -1 and 1)
    for i, idx in enumerate(control_indices):
        control_points[idx] = trial.suggest_float(f'control_point_{i}', -10.0, 10.0)
    
    # Generate pattern from control points
    new_pattern = bezier_curve(control_points, num_points=20)
    
    # Select activation function
    activation = trial.suggest_categorical('activation', 
                                         ['relu', 'linear', 'tanh', 'sigmoid', 'quadratic', 'softplus'])
    
    # Create a temporary copy of pattern list
    temp_pattern_list = current_patterns.copy()
    temp_pattern_starts = current_starts.copy()
    temp_pattern_ends = current_ends.copy()
    
    # Update or add new pattern
    if pattern_index < len(temp_pattern_list):
        temp_pattern_list[pattern_index] = new_pattern
        temp_pattern_starts[pattern_index] = pattern_start
        temp_pattern_ends[pattern_index] = pattern_end
    else:
        temp_pattern_list.append(new_pattern)
        temp_pattern_starts.append(pattern_start)
        temp_pattern_ends.append(pattern_end)
    
    # Generate features using the pattern correlations
    train_features = calculate_correlations_as_features(
        X_train_standardized[sample_indices], 
        temp_pattern_list, 
        temp_pattern_starts, 
        temp_pattern_ends,
        activation
    )
    
    # Create validation split for early stopping
    X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
        train_features, 
        y_train.iloc[sample_indices], 
        test_size=0.2, 
        random_state=42
    )
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(**get_lgb_params())
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    model.fit(
        X_train_lgb, 
        y_train_lgb, 
        eval_set=[(X_val_lgb, y_val_lgb)],
        callbacks=callbacks,
        eval_metric='auc'
    )
    
    # Get LightGBM predictions on validation set
    val_preds = model.predict_proba(X_val_lgb)[:, 1]
    
    # Calculate AUC
    return roc_auc_score(y_val_lgb, val_preds)

class PatternOptimizer:
    """
    Class for optimizing patterns using Bezier curves and LightGBM
    """
    def __init__(self, X_train_standardized, y_train, X_test_standardized, y_test):
        """
        Initialize the pattern optimizer
        
        Parameters:
        -----------
        X_train_standardized : array-like
            Standardized training data
        y_train : array-like
            Training labels
        X_test_standardized : array-like
            Standardized test data
        y_test : array-like
            Test labels
        """
        self.X_train_standardized = X_train_standardized
        self.y_train = y_train
        self.X_test_standardized = X_test_standardized
        self.y_test = y_test
        
        # Initialize patterns and correlations tracking
        self.pattern_list = []
        self.pattern_starts = []
        self.pattern_ends = []
        self.pattern_control_points = []  # Store control points for each pattern
        self.best_activations = []  # Track activations for each pattern
        
        # Tracking variables
        self.current_val_auc = 0
        self.current_train_auc = 0
        self.iteration = 0
        self.best_test_auc = 0
        self.best_model = None
        self.best_pattern_list = None
        self.best_pattern_starts = None
        self.best_pattern_ends = None
        self.best_activations_copy = []
        self.best_pattern_control_points = []
        
        # Stagnation and adaptive n_trials parameters
        self.stagnation_counter = 0
        self.best_overall_val_auc = 0.0  # Track the best validation AUC achieved
        self.base_n_trials = 500       # Base number of Optuna trials
        self.current_n_trials = self.base_n_trials  # Current number of trials, adapted during stagnation
    
    def run_optimization(self, max_iterations=10, visualize_func=None):
        """
        Run pattern optimization for multiple iterations
        
        Parameters:
        -----------
        max_iterations : int, optional
            Maximum number of iterations to run
        visualize_func : function, optional
            Function to call for visualization of patterns
            Expected signature: func(iteration, patterns, starts, ends, activations, model, X_train_std, **kwargs)
            
        Returns:
        --------
        model : trained model
            The final trained model
        patterns : list of arrays
            The final optimized patterns
        starts : list of int
            Start indices for each pattern
        ends : list of int
            End indices for each pattern
        activations : list of str
            Activation types for each pattern
        control_points : list of arrays
            Control points for each pattern
        """
        while self.iteration < max_iterations:
            # Sample 70% of training data for optimization
            sample_size = int(len(self.y_train) * 0.7)
            sample_indices = np.random.choice(len(self.y_train), sample_size, replace=False)
            
            # Create Optuna study with pruning for faster convergence
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
            )
            
            # Create a partial function with fixed sample_indices
            objective = lambda trial: objective_function(
                trial, sample_indices, 
                self.X_train_standardized, self.y_train,
                self.pattern_list, self.pattern_starts, self.pattern_ends
            )
            
            # Run optimization
            optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce verbosity
            # Use current_n_trials which adapts based on stagnation
            study.optimize(objective, n_trials=self.current_n_trials)
            
            # Extract best parameters
            best_params = study.best_params
            pattern_index_float = best_params.get('pattern_index', 0)
            pattern_start = best_params.get('pattern_start', 0)
            pattern_end = best_params.get('pattern_end', 19)
            activation = best_params.get('activation', 'relu')
            
            # Convert index to integer and bound it
            max_index = len(self.pattern_list)
            pattern_index = min(max_index, math.floor(pattern_index_float * (max_index + 1)))
            
            # Create new pattern from best parameters
            control_points = [0.0] * 20  # Initialize with zeros
            active_region_size = pattern_end - pattern_start + 1
            control_indices = [
                pattern_start,  # First control point at start
                pattern_start + active_region_size // 3,  # Second at 1/3
                pattern_start + 2 * active_region_size // 3,  # Third at 2/3
                pattern_end  # Last at end
            ]
            
            # Set control point values from best params (-1 to 1 range)
            for i, idx in enumerate(control_indices):
                control_points[idx] = best_params.get(f'control_point_{i}', 0.0)
            
            # Generate pattern from control points
            new_pattern = bezier_curve(control_points, num_points=20)
            
            # Action type
            action = "Updating" if pattern_index < len(self.pattern_list) else "Creating new"
            
            # Update or add new pattern and activation
            if pattern_index < len(self.pattern_list):
                self.pattern_list[pattern_index] = new_pattern
                self.pattern_starts[pattern_index] = pattern_start
                self.pattern_ends[pattern_index] = pattern_end
                self.pattern_control_points[pattern_index] = control_points.copy()  # Store control points
                if len(self.best_activations) <= pattern_index:
                    self.best_activations.append(activation)
                else:
                    self.best_activations[pattern_index] = activation
            else:
                self.pattern_list.append(new_pattern)
                self.pattern_starts.append(pattern_start)
                self.pattern_ends.append(pattern_end)
                self.pattern_control_points.append(control_points.copy())  # Store control points
                self.best_activations.append(activation)
            
            # Train and evaluate model with updated patterns
            model, self.current_train_auc, test_auc, self.current_val_auc = train_and_evaluate(
                self.X_train_standardized, self.y_train,
                self.X_test_standardized, self.y_test,
                self.pattern_list, self.pattern_starts, self.pattern_ends, self.best_activations
            )
            
            # Track best performance
            if test_auc > self.best_test_auc:
                self.best_test_auc = test_auc
                self.best_model = model
                self.best_pattern_list = self.pattern_list.copy()
                self.best_pattern_starts = self.pattern_starts.copy()
                self.best_pattern_ends = self.pattern_ends.copy()
                self.best_activations_copy = self.best_activations.copy()
                self.best_pattern_control_points = self.pattern_control_points.copy()
            
            print(f"Iteration {self.iteration+1}, {action} pattern at index {pattern_index}, Activation: {activation}, Range: [{pattern_start}:{pattern_end}], Train AUC: {self.current_train_auc:.4f}, Val AUC: {self.current_val_auc:.4f}, Test AUC: {test_auc:.4f}")
            
            # Call visualization function if provided
            if visualize_func is not None:
                visualize_func(
                    self.iteration,
                    list(self.pattern_list),
                    list(self.pattern_starts),
                    list(self.pattern_ends),
                    list(self.best_activations),
                    self.best_model,
                    self.X_train_standardized,
                    output_dir=f'plots',
                    control_points=list(self.pattern_control_points) if hasattr(self, 'pattern_control_points') else None
                )
            
            # Increment iteration counter
            self.iteration += 1
            
            # --- Stagnation and adaptive n_trials logic ---
            if self.current_val_auc > self.best_overall_val_auc:
                self.best_overall_val_auc = self.current_val_auc
                self.stagnation_counter = 0
                self.current_n_trials = self.base_n_trials  # Reset n_trials
            else:
                self.stagnation_counter += 1
                self.current_n_trials *= 2  # Double n_trials

            # Check stopping condition based on stagnation
            if self.stagnation_counter >= 3:
                print(f"Stopping optimization after {self.iteration} iterations due to stagnation for 3 consecutive iterations.")
                break
            # --- End Stagnation Logic ---
        
        # Final evaluation with best model
        final_test_features = calculate_correlations_as_features(
            self.X_test_standardized, self.best_pattern_list, 
            self.best_pattern_starts, self.best_pattern_ends, self.best_activations_copy
        )
        
        # Use the best model for final predictions
        final_test_preds = self.best_model.predict_proba(final_test_features)[:, 1]
        final_test_auc = roc_auc_score(self.y_test, final_test_preds)

        print(f"Test AUC with LightGBM: {final_test_auc:.4f}")
        print(f"Number of patterns used in best model: {len(self.best_pattern_list)}")
        
        return (self.best_model, self.best_pattern_list, self.best_pattern_starts, 
                self.best_pattern_ends, self.best_activations_copy, 
                self.best_pattern_control_points) 