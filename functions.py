import numpy as np
import matplotlib.pyplot as plt

def get_match_counts(params, X):
    pattern_length = int(params[0])
    start = int(params[1]) 
    end = int(params[2])
    stride = int(params[3])
    selected_area = X[:, start:end:stride]
    cct = params[4]
    pattern = np.array(params[5:(5 + pattern_length)])
    
    # Ensure valid dimensions
    window_size = len(pattern) * stride
    n_windows = max(0, (selected_area.shape[1] - window_size + stride) // stride)
    
    if n_windows <= 0:
        return np.zeros((X.shape[0], 1))
        
    # Create strided view of the data for efficient sliding window
    shape = (selected_area.shape[0], n_windows, len(pattern))
    strides = (selected_area.strides[0], selected_area.strides[1] * stride, selected_area.strides[1])
    windows = np.lib.stride_tricks.as_strided(selected_area, shape=shape, strides=strides)
    
    # Compute correlations efficiently using matrix operations
    means = windows.mean(axis=2, keepdims=True)
    stds = windows.std(axis=2, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero
    normalized_windows = (windows - means) / stds
    pattern_norm = (pattern - pattern.mean()) / (pattern.std() or 1)
    correlations = np.abs(np.sum(normalized_windows * pattern_norm, axis=2) / len(pattern))
    
    return np.sum(correlations > cct, axis=1, keepdims=True)

def visualize_pattern(params):
    pattern_length = int(params[0])
    pattern = np.array(params[5:(5 + pattern_length)])
    
    plt.figure(figsize=(10,4))
    plt.plot(pattern, label='Pattern')
    plt.title('Discovered Pattern')
    plt.xlabel('Time')
    plt.ylabel('Value') 
    plt.legend()
    plt.show()

def locate_pattern(params, X, row):
    pattern_length = int(params[0])
    start = int(params[1]) 
    end = int(params[2])
    stride = int(params[3])
    selected_area = X[row, start:end:stride]
    cct = params[4]
    pattern = np.array(params[5:(5 + pattern_length)])
    
    # Create windows
    window_size = len(pattern) * stride
    n_windows = max(0, (len(selected_area) - window_size + stride) // stride)
    
    if n_windows <= 0:
        return [], []
        
    windows = np.lib.stride_tricks.as_strided(
        selected_area, 
        shape=(n_windows, len(pattern)),
        strides=(selected_area.strides[0] * stride, selected_area.strides[0])
    )
    
    # Compute correlations
    means = windows.mean(axis=1, keepdims=True)
    stds = windows.std(axis=1, keepdims=True)
    stds[stds == 0] = 1
    normalized_windows = (windows - means) / stds
    pattern_norm = (pattern - pattern.mean()) / (pattern.std() or 1)
    correlations = np.abs(np.sum(normalized_windows * pattern_norm, axis=1) / len(pattern))
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot signal
    ax1.plot(selected_area, label='Signal')
    matches = np.where(correlations > cct)[0]
    ax1.set_title(f'Signal with Pattern Matches (Count: {len(matches)})')
    ax1.legend()
    
    # Plot correlations
    ax2.plot(correlations, label='Correlation')
    ax2.axhline(y=cct, color='r', linestyle='--', label='Threshold')
    ax2.fill_between(range(len(correlations)), correlations, cct, 
                    where=(correlations > cct), color='green', alpha=0.3)
    ax2.set_title('Correlation Coefficient')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return matches, correlations