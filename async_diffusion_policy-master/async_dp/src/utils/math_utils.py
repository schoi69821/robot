import numpy as np

def linear_interpolate(p_start, p_end, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    return p_start + (p_end - p_start) * alpha

def apply_ema_filter(curr, prev, alpha=0.2):
    if prev is None: return curr
    return (curr * alpha) + (prev * (1.0 - alpha))

def normalize_data(data, stats):
    # Normalize to [-1, 1]
    range_val = stats['max'] - stats['min']
    if isinstance(range_val, (int, float)):
        if abs(range_val) < 1e-6:
            return data * 0
    else:
        range_val = np.where(np.abs(range_val) < 1e-6, 1.0, range_val)
    return (data - stats['min']) / range_val * 2 - 1

def unnormalize_data(data, stats):
    # Restore from [-1, 1]
    data = (data + 1) / 2
    return data * (stats['max'] - stats['min']) + stats['min']

def get_stats(dataset):
    # Dummy stats for dry-run
    return {'min': -3.14, 'max': 3.14}