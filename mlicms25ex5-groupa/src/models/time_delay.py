from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import RegressorMixin

@dataclass
class TimeDelayEmbedding(RegressorMixin):
    """Transformer that creates a time-delay embedding of time-series data.
    
    The class takes a time series and constructs a new feature set by 
    appending delayed versions of the input, specified by the provided time delays.
    
    Parameters:
    -----------
        time_delay: int | list[int] | npt.ArrayLike
            One or more delay steps to include in the embedding.
            If an integer k is given, it is expanded to delays [1, 2, ..., k].

    """


    time_delay: npt.ArrayLike | int | list[int]

    def __post_init__(self):
        # We allow 'time_delay' to be int or list when
        # creating the object. But we need to transform
        # it to the np.array then.
        if isinstance(self.time_delay, int):
            self.time_delay = np.arange(1, self.time_delay + 1)
        elif isinstance(self.time_delay, list):
            self.time_delay = np.asarray(self.time_delay)
        else:
            self.time_delay = np.asarray(self.time_delay)
            
        # Ensure it's a numpy array at this point
        assert isinstance(self.time_delay, np.ndarray)

    def transform(self, x: npt.ArrayLike) -> np.ndarray:
        """Computes a time-delay embedding of the provided data.

        The delays are defined by self.time_delay, and the
        last max(self.time_delay) of datapoints are discarded.

        Parameters:
        -----------
        x : npt.ArrayLike, shape (N,) or (N, d)
            Input data.

        Returns:
        --------
        np.ndarray, shape (N - max_delay, d + d * len(self.time_delay))
            Time-delayed data. If d > 1, the embedding is flattened
            for each data point. The first d columns of the output array
            are the original data points.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]

        time_delay_array = np.asarray(self.time_delay, dtype=int)
        max_delay = time_delay_array.max()

        # # Handle 1D input by reshaping to (N, 1)
        # if x.ndim == 1:
        #     x = x[:, np.newaxis]

        N, d = x.shape
        embedded_length = N - max_delay

        if embedded_length <= 0:
            raise ValueError("Signal too short for given delays")

        # Start with original data (undelayed)
        # embedded = x[max_delay:N]
        cols = []
        # Add delayed versions
        for delay in sorted(time_delay_array, reverse=True):
            cols.append(x[max_delay - delay : N - delay])
        cols.append(x[max_delay:N])  # Add the undelayed data

        return np.hstack(cols)



def create_time_delay_embedding(signal, delay, embed_dim):
    """
    Create time delay embedding of a signal
    
    Parameters:
    signal: 1D array of the time series
    delay: time delay (number of samples)
    embed_dim: embedding dimension
    
    Returns:
    embedded: array of shape (N-delay*(embed_dim-1), embed_dim)
    """
    signal = np.asarray(signal)
    N = len(signal)
    embedded_length = N - delay * (embed_dim - 1)
    
    if embedded_length <= 0:
        raise ValueError("Signal too short for given delay and embedding dimension")
    
    embedded = np.zeros((embedded_length, embed_dim))
    
    for i in range(embed_dim):
        start_idx = i * delay
        end_idx = start_idx + embedded_length
        embedded[:, i] = signal[start_idx:end_idx]
    
    return embedded


def analyze_embedding_quality(signal, delay, embed_dims):
    """
    Analyze the quality of different embeddings
    
    Parameters:
    signal: 1D array of the time series
    delay: time delays to test
    embed_dims: list of embedding dimensions to test
    
    Returns:
    pd.DataFrame: DataFrame with quality metrics for each combination
    """
    results = []
    
    for d in delay:
        for embed_dim in embed_dims:
            try:
                embedded = create_time_delay_embedding(signal, d, embed_dim)
                
                # Calculate some quality metrics
                # 1. Variance of each dimension
                variances = np.var(embedded, axis=0)
                total_variance = np.sum(variances)
                
                # 2. Correlation between dimensions (should be low for good embedding)
                corr_matrix = np.corrcoef(embedded.T)
                # Average off-diagonal correlation (absolute value)
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                avg_correlation = np.mean(np.abs(corr_matrix[mask]))
                
                results.append({
                    'delay': d,
                    'embed_dim': embed_dim,
                    'total_variance': total_variance,
                    'avg_correlation': avg_correlation,
                    'embedding_size': embedded.shape[0]
                })
                
            except ValueError:
                continue
    
    return pd.DataFrame(results)
        

