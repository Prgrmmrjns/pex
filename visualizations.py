import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import comb

# Add Bezier curve function for visualization
def bernstein_poly(i, n, t):
    """Bernstein polynomial basis function"""
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, num_points=100):
    """Generate a Bezier curve from control points"""
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros(num_points)
    
    for i, point in enumerate(control_points):
        curve += point * np.array([bernstein_poly(i, n, t_) for t_ in t])
    
    return curve

def visualize_pattern(pattern, start, end, activation_type, idx, X_standardized, y_train=None, output_dir='plots', control_points=None):
    """
    Create a single visualization with pattern on the left and class-based visualization on the right
    
    Parameters:
    -----------
    pattern : array-like
        The pattern values to visualize
    start : int
        Start index of the active region
    end : int
        End index of the active region
    activation_type : str
        Name of the activation function used
    idx : int
        Index of the pattern (used for filename)
    X_standardized : array-like
        Standardized input data for activation visualization
    y_train : array-like, optional
        Target class labels for creating class-based visualizations
    output_dir : str, optional
        Directory to save plots, default is 'plots'
    control_points : array-like, optional
        Control points for the Bezier curve (if available)
    """
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Only show the active region of the pattern
    active_x = np.arange(start, end + 1)
    active_pattern = pattern[start:end + 1]
    
    # Plot only the active region of the pattern
    ax1.plot(active_x, active_pattern, 'b-', label='Pattern', linewidth=2)
    
    # Set x-axis limits to focus on active region
    padding = max(1, (end - start) * 0.1)  # Add 10% padding
    ax1.set_xlim(start - padding, end + padding)
    
    # Mark start and end points
    ax1.scatter([start, end], [pattern[start], pattern[end]], color='red', s=100, zorder=5, 
                label='Active Region Boundaries')
    
    # Plot control points if provided
    if control_points is not None:
        # Find non-zero values in control points within active region
        control_x = []
        control_y = []
        for i, val in enumerate(control_points):
            if val != 0 and start <= i <= end:
                control_x.append(i)
                control_y.append(val)
        
        # Plot control points
        ax1.scatter(control_x, control_y, color='green', s=100, zorder=10, label='Control Points')
        
        # Connect control points with a dashed line
        ax1.plot(control_x, control_y, 'g--', alpha=0.5, zorder=5)
    
    ax1.set_title(f'Pattern {idx} (Active Region Only)')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Value')
    
    # Set integer ticks for the x-axis
    tick_step = max(1, (end - start) // 5)  # Show at most 5-6 ticks in active region
    x_ticks = range(start, end + 1, tick_step)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(i) for i in x_ticks])
    
    ax1.grid(True)
    ax1.legend()
    
    # Prepare the pattern for correlation calculation
    pattern_mean = np.mean(pattern[start:end+1])
    pattern_std_dev = np.std(pattern[start:end+1]) or 1
    
    pattern_std = pattern.copy()
    pattern_std[start:end+1] = (pattern[start:end+1] - pattern_mean) / pattern_std_dev
    
    # Zero out pattern values outside the start-end range
    pattern_mask = np.zeros_like(pattern_std)
    pattern_mask[start:end+1] = 1
    pattern_std = pattern_std * pattern_mask
    
    # Calculate raw correlations
    raw_correlations = np.sum(X_standardized * pattern_std, axis=1) / (end - start + 1)
    
    # Apply the activation function
    if activation_type == 'relu':
        activations = np.maximum(0, raw_correlations)
    elif activation_type == 'linear':
        activations = raw_correlations
    elif activation_type == 'tanh':
        activations = np.tanh(raw_correlations)
    elif activation_type == 'sigmoid':
        activations = 1 / (1 + np.exp(-raw_correlations))
    elif activation_type == 'quadratic':
        activations = raw_correlations ** 2
    elif activation_type == 'softplus':
        activations = np.log1p(np.exp(raw_correlations))
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")
    
    # Plot class-based visualization if target labels are provided
    if y_train is not None and len(y_train) == len(X_standardized):
        # Get unique classes
        classes = np.unique(y_train)
        colors = ['blue', 'red']  # For binary classification
        
        # Create class distribution by correlation strength
        ax2.set_title(f'Pattern Activated Correlation Distribution by Class')
        ax2.set_xlabel('Activated Correlation Value')
        ax2.set_ylabel('Density')
        
        for i, cls in enumerate(classes):
            class_mask = (y_train == cls)
            # Use activated values for plotting
            class_activated_values = activations[class_mask] 
            
            # Plot histogram or KDE for each class using activated values
            ax2.hist(class_activated_values, bins=30, density=True, alpha=0.5, 
                    color=colors[i], label=f'Class {cls}')
            
        ax2.legend()
        ax2.grid(True)
    else:
        # Alternative visualization: Correlation heatmap by percentile
        # Create 5 bins of activation strength (using activated values)
        num_bins = 5
        # Use activated values for percentile calculation
        activation_bins = np.percentile(activations, np.linspace(0, 100, num_bins+1)) 
        # Handle potential edge case where all activations are the same
        if len(np.unique(activation_bins)) < num_bins + 1:
            # Use linspace if percentiles produce duplicate bins (e.g., all zeros after ReLU)
             activation_bins = np.linspace(activations.min(), activations.max(), num_bins + 1)
             
        # Ensure bins are monotonic
        activation_bins = np.unique(activation_bins) 
        
        if len(activation_bins) < 2: # Handle case where all activations are identical
            bin_indices = np.zeros(len(activations), dtype=int)
            num_bins = 1 # Only one bin
        else:
            bin_indices = np.digitize(activations, activation_bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(activation_bins) - 2) # Clip to valid bin indices
            num_bins = len(activation_bins) -1 # Update num_bins based on unique bins
        
        # For each bin, compute average feature values
        bin_feature_means = np.zeros((num_bins, X_standardized.shape[1]))
        for bin_idx in range(num_bins):
            mask = (bin_indices == bin_idx)
            if np.any(mask):
                bin_feature_means[bin_idx] = X_standardized[mask].mean(axis=0)
        

        feature_means_active = bin_feature_means[:, start:end+1]
        
        # Plot heatmap
        im = ax2.imshow(feature_means_active, aspect='auto', cmap='coolwarm')
        ax2.set_title('Avg Feature Values by Activation Percentile')
        ax2.set_xlabel('Feature Index (Active Region)')
        ax2.set_ylabel('Activation Strength (Percentile)')
        
        # Set y-axis labels as percentiles
        # Adjust label generation if num_bins changed
        if num_bins > 0:
            y_labels = [f"{int(p)}%" for p in np.linspace(0, 100, num_bins)]
            ax2.set_yticks(np.arange(num_bins))
            ax2.set_yticklabels(y_labels)
        else: # Handle case with only one bin
            ax2.set_yticks([0])
            ax2.set_yticklabels(["100%"])

        # Set x-axis ticks
        x_ticks_heatmap = np.linspace(0, end-start, min(5, end-start+1)).astype(int)
        x_tick_labels = [str(start + i) for i in x_ticks_heatmap]
        ax2.set_xticks(x_ticks_heatmap)
        ax2.set_xticklabels(x_tick_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Average Feature Value')
    
    # Set figure title
    fig.suptitle(f'Pattern {idx} with {activation_type.capitalize()} Activation')
    plt.tight_layout()

    plt.savefig(f'plots/pattern_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_iteration_results(iteration, patterns, starts, ends, activations, model, X_train_std, y_train=None, output_dir='plots', control_points=None, pattern_index=None):
    """
    Create visualizations for patterns in the current iteration
    
    Parameters:
    -----------
    iteration : int or str
        Current iteration number (used for file path)
    patterns : list of arrays
        List of patterns to visualize
    starts : list of int
        List of start indices for each pattern
    ends : list of int
        List of end indices for each pattern
    activations : list of str or list
        Activation type(s) for patterns
    model : trained model
        The trained model (not used for visualization)
    X_train_std : array-like
        Standardized training data
    y_train : array-like, optional
        Target class labels for visualization
    output_dir : str, optional
        Directory to save plots, default is 'plots'
    control_points : list of arrays, optional
        Control points for Bezier curves (if available)
    pattern_index : int, optional
        Index of the pattern in the global pattern list (if known)
    """
    
    # Handle activations as string or list
    if isinstance(activations, str):
        act_list = [activations] * len(patterns)
    else:
        act_list = activations
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each pattern with a single image
    for i, (pattern, start, end, activation) in enumerate(zip(patterns, starts, ends, act_list)):
        # Get control points for this pattern if available
        cp = None
        if control_points is not None and i < len(control_points):
            cp = control_points[i]
        
        # Use pattern_index if provided, otherwise use local index
        idx = pattern_index if pattern_index is not None else i
        
        visualize_pattern(
            pattern, start, end, activation, idx, 
            X_train_std, y_train, output_dir, cp
        ) 