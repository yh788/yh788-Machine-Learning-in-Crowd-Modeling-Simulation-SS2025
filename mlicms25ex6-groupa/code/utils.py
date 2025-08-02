import keras.losses
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from tqdm import tqdm
import pandas as pd


def weidmann(sk, v0, T, l):
    """
    Computes the pedestrian speed based on the Weidmann model for the fundamental diagram.

    The model describes pedestrian dynamics by expressing walking speed
    as a function of local density (sk) to the K nearest neighbors.

    Parameters
    ----------
    sk : float
        Average spacing of the K nearest neighbors.
    v0 : float
        Desired walking speed of the pedestrian (m/s).
    T : float
        Time gap between pedestrians (s).
    l : float
        Size of a pedestrian (m).

    Returns
    -------
    float
        The resulting pedestrian speed computed from the Weidmann model.
    """
    return v0 * (1 - np.exp((l - sk) / (v0 * T)))


def build_model(layers, input_dimension, learning_rate, activation="relu"):
    """
    Builds and compiles a feed-forward neural network model.

    Creates a Sequential neural network model with the specified number of fully connected  hidden layers.
    Each hidden layer uses ReLU activation. The final output layer is a single neuron.

    Parameters
    ----------
    layers : list of int, optional
        List specifying the number of units in each hidden layer. Default is [32].
        Example: [64, 32] will create two hidden layers with 64 and 32 units.
    input_dimension : int
        The number of input features.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    nn_model : keras.models.Sequential
        A compiled Keras Sequential model to train.
    """

    nn_model = Sequential()

    # Add first layer
    nn_model.add(
        Dense(layers[0], activation=activation, input_shape=(input_dimension,))
    )

    # Add remaining hidden layers
    for layer in layers[1:]:
        nn_model.add(Dense(layer, activation=activation))

    # Output layer
    nn_model.add(Dense(1))

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Compile
    nn_model.compile(
        optimizer=optimizer,
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_absolute_error],
    )

    return nn_model


def neural_network(
    X_train,
    y_train,
    n_splits=5,
    layers=[32],
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    verbose=1,
    n_bootstrap=50,
    activation="relu",
):
    """
    A feed-forward neural network to predict pedestrian speed.

    The model input consists of 2K + 1 features:
    - The mean spacing to the K nearest neighbors.
    - The relative positions (xi - x, yi - y) of the K neighbors.

    Parameters
    ----------
    X_train : np.ndarray,
        Input training data containing average spacing and K relative positions.
    y_train : np.ndarray
        Target speed values for each pedestrian.
    n_splits : int
        Number of K-Fold splits for validation.
    layers : list of int, optional
        List specifying the number of units in each hidden layer. Default is [32].
        Example: [64, 32] will create two hidden layers with 64 and 32 units.
    epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Size of the training batches. Default is 32.
    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is 0.001.
    verbose : int, optional
        Verbosity level for training. 0 = silent, 1 = progress bar. Default is 1.

    Returns
    -------
    model : tf.keras.Model
        The trained Keras Sequential model.
    history : tf.keras.callbacks.History
        The training history object containing loss values and metrics.

    Example:
    -------
    # X_train shape: (N, 2K+1), y_train shape: (N,)
    model, history = neural_network(
        X_train, y_train,
        layers=[128, 64, 32],  # 3 hidden layers
        epochs=150,
        learning_rate=0.0005
    )
    """

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"--- Training fold {fold + 1}/{n_splits} ---")

        # Build model for each fold
        model = build_model(
            layers=layers,
            input_dimension=X_train.shape[1],
            learning_rate=learning_rate,
            activation=activation,
        )

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        if n_bootstrap > 0:
            X_bootstrap_all = []
            y_bootstrap_all = []
            for _ in range(n_bootstrap):
                X_resampled, y_resampled = resample(
                    X_train_fold, y_train_fold, replace=True
                )
                X_bootstrap_all.append(X_resampled)
                y_bootstrap_all.append(y_resampled)

            X_train_for_fit = np.concatenate(X_bootstrap_all)
            y_train_for_fit = np.concatenate(y_bootstrap_all)
            print(
                f"Bootstrap sampling created a training set of size {len(X_train_for_fit)}"
            )
        else:
            X_train_for_fit = X_train_fold
            y_train_for_fit = y_train_fold

        model.fit(
            X_train_for_fit,
            y_train_for_fit,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(X_val_fold, y_val_fold),
        )

        val_loss, val_mae = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f"Fold {fold + 1} MSE: {val_loss:.4f}, MAE: {val_mae:.4f}")
        scores.append(val_loss)

    print(f"\nAverage K-Fold MSE: {np.mean(scores):.4f}")

    return model, scores


def process_dataset(df, K=10):
    """

    This function extracts pedestrian positions and velocities from a DataFrame containing time-ordered
    frames of pedestrian data. It computes:
    - The mean spacing to the K nearest neighbors for each pedestrian.
    - The relative positions of those K neighbors.
    - The pedestrian speed as the Euclidean distance between consecutive frames.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with pedestrian data. Contains columns:
        - 'frame': integer frame number
        - 'id': pedestrian identifier
        - 'x', 'y', 'z': pedestrian coordinates at that frame
    K : int, optional
        Number of nearest neighbors to consider. Default is 10.

    Returns
    -------
    dataset_weidmann : list of tuple(float, float)
        List of (mean_spacing, speed) pairs for use in the Weidmann model.

    dataset_neural_network : list of tuple(list[float], float)
        List of (input_features, speed) pairs for training a neural network.
        input_features = [mean_spacing, dx1, dy1, ..., dxK, dyK],
        where (dxi, dyi) are the relative positions of neighbor i.

    Notes
    -----
    - Only frames with more than 10 pedestrians are considered.
    - Speeds are computed as ||pos_t+1 - pos_t||.
    - KDTree is used for efficient neighbor lookup.
    """

    dataset_weidmann = []  # X: mean spacing, y: speed
    dataset_neural_network = []  # X: mean spacing + K relative positions, y: speed

    # Make sure each frame has at least 10 pedestrians
    grouped = df.groupby("frame")
    frames = sorted([frame for frame, group in grouped if len(group) > 10])
    filtered_df = df[df["frame"].isin(frames)]
    grouped = filtered_df.groupby("frame")

    # For each frame
    for i in range(len(frames) - 1):
        # Get frames t and t+1
        frame_t = frames[i]
        frame_tp1 = frames[i + 1]

        # Get agent positions at t and t+1
        agents_t = grouped.get_group(frame_t).set_index("id")
        agents_tp1 = grouped.get_group(frame_tp1).set_index("id")

        # Get position values of pedestrian the pedestrian (x, y) if the pedestrians exists at both frames t and t+1
        common_ids = agents_t.index.intersection(agents_tp1.index)
        pos_t = agents_t.loc[common_ids, ["x", "y"]].values
        pos_tp1 = agents_tp1.loc[common_ids, ["x", "y"]].values
        ids = common_ids.values

        # Build KDTree for fast neighbor search
        tree = KDTree(pos_t)
        dists, idxs = tree.query(
            pos_t, k=K + 1
        )  # include self in K+1, will discard later

        for j, agent_id in enumerate(ids):
            # Calculate speed, multiply by 16 for frame rate
            vi = (
                np.linalg.norm(pos_tp1[j] - pos_t[j]) * 16
            )  # speed = ||x_t+1 - x_t|| * 16

            neighbors_idx = idxs[j][1 : K + 1]  # exclude self
            neighbors = pos_t[neighbors_idx]

            rel_pos = (
                neighbors - pos_t[j]
            )  # relative positions (xi - x, yi - y)
            mean_spacing = dists[j][
                1 : K + 1
            ].mean()  # mean spacing without self

            # For Weidmann: X = mean spacing, y = speed
            dataset_weidmann.append((mean_spacing, vi))

            # For NN: X = [mean_spacing, dx1, dy1, ..., dxK, dyK], y = speed
            nn_input = [mean_spacing] + rel_pos.flatten().tolist()
            dataset_neural_network.append((nn_input, vi))

    return dataset_weidmann, dataset_neural_network


def train_with_bootstrap_validation(
    X_train_full,
    y_train_full,
    X_test,
    y_test,
    layers,
    epochs,
    batch_size,
    learning_rate,
    activation,
    n_bootstrap=50,
    verbose=0,
):
    """
    Trains a neural network using bootstrapped training sets and evaluates on a fixed test set.

    This function repeatedly resamples the training set with replacement to create
    multiple training sets. For each bootstrap iteration, a new model is trained,
    and MSE is recorded on both the  resampled train set and the fixed test set.

    Parameters
    ----------
    X_train_full : np.ndarray
        Full input training features.
    y_train_full : np.ndarray
        Full target values corresponding to X_train_full.
    X_test : np.ndarray
        Test input features used for evaluation.
    y_test : np.ndarray
        Test target values used for evaluation.
    layers : list of int
        List specifying the number of units in each hidden layer.
    epochs : int
        Number of epochs for training in each bootstrap iteration.
    batch_size : int
        Size of mini-batches used during training.
    learning_rate : float
        Learning rate for the Adam optimizer.
    activation : str
        Activation function used in hidden layers (e.g., 'relu', 'tanh').
    n_bootstrap : int, optional
        Number of bootstrap samples to use. Default is 50.
    verbose : int, optional
        Verbosity level for training. 0 = silent, 1 = progress messages.

    Returns
    -------
    train_mse_mean : float
        Mean of training MSEs across bootstrap iterations.
    train_mse_std : float
        Standard deviation of training MSEs across bootstrap iterations.
    test_mse_mean : float
        Mean of test MSEs across bootstrap iterations.
    test_mse_std : float
        Standard deviation of test MSEs across bootstrap iterations.
    """
    test_scores = []
    train_scores = []

    for i in tqdm(range(n_bootstrap), desc=f"Bootstrap Iterations"):
        X_resampled, y_resampled = resample(
            X_train_full, y_train_full, replace=True
        )

        model = build_model(
            layers=layers,
            input_dimension=X_train_full.shape[1],
            learning_rate=learning_rate,
            activation=activation,
        )

        model.fit(
            X_resampled,
            y_resampled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Get both train and test errors
        train_loss, _ = model.evaluate(X_resampled, y_resampled, verbose=0)
        test_loss, _ = model.evaluate(X_test, y_test, verbose=0)

        train_scores.append(train_loss)
        test_scores.append(test_loss)

    return (
        np.mean(train_scores),
        np.std(train_scores),
        np.mean(test_scores),
        np.std(test_scores),
    )


def evaluate_nn(
    dataset_corridor_nn,
    dataset_bottleneck_nn,
    sample_size=10000,
    layers=[3],
    epochs=5,
    batch_size=256,
    learning_rate=0.01,
    activation="relu",
    n_bootstrap=10,
    verbose=0,
):
    """
    Evaluates neural network performance across multiple training/testing scenarios using bootstrapped validation.

    This function compares the neural network using different  datasets in multiple training/testing
    combinations. Datasets can optionally be subsampled for faster evaluation. Results include the
    mean and standard deviation of MSE for both training and test sets over multiple bootstrap iterations.

    Parameters
    ----------
    dataset_corridor_nn : list of tuple(list[float], float)
        List of (input_features, speed) samples from the corridor dataset.
    dataset_bottleneck_nn : list of tuple(list[float], float)
        List of (input_features, speed) samples from the bottleneck dataset.
    sample_size : int, optional
        Maximum number of samples per dataset to speed up training. Default is 10,000.
    layers : list of int, optional
        Number of units in each hidden layer. Default is [3].
    epochs : int, optional
        Number of epochs per training iteration. Default is 5.
    batch_size : int, optional
        Mini-batch size for training. Default is 256.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    activation : str, optional
        Activation function used in hidden layers. Default is 'relu'.
    n_bootstrap : int, optional
        Number of bootstrap iterations for model training. Default is 10.
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = detailed training logs.

    Returns
    -------
    nn_results_df : pandas.DataFrame
        DataFrame containing evaluation metrics for each scenario, including:
        - Train/Test dataset names
        - Mean/Std MSE on training and test sets
        - Sample sizes for train/test sets
    """
    # Extract full features and targets
    corridor_nn_data_full = np.array([x for x, _ in dataset_corridor_nn])
    corridor_nn_targets_full = np.array([y for _, y in dataset_corridor_nn])
    bottleneck_nn_data_full = np.array([x for x, _ in dataset_bottleneck_nn])
    bottleneck_nn_targets_full = np.array(
        [y for _, y in dataset_bottleneck_nn]
    )

    # Sample corridor data
    if len(corridor_nn_data_full) > sample_size:
        np.random.seed(42)  # For reproducible results
        corridor_indices = np.random.choice(
            len(corridor_nn_data_full), sample_size, replace=False
        )
        corridor_nn_data = corridor_nn_data_full[corridor_indices]
        corridor_nn_targets = corridor_nn_targets_full[corridor_indices]
    else:
        corridor_nn_data = corridor_nn_data_full
        corridor_nn_targets = corridor_nn_targets_full

    # Sample bottleneck data
    if len(bottleneck_nn_data_full) > sample_size:
        np.random.seed(43)  # Different seed for different sampling
        bottleneck_indices = np.random.choice(
            len(bottleneck_nn_data_full), sample_size, replace=False
        )
        bottleneck_nn_data = bottleneck_nn_data_full[bottleneck_indices]
        bottleneck_nn_targets = bottleneck_nn_targets_full[bottleneck_indices]
    else:
        bottleneck_nn_data = bottleneck_nn_data_full
        bottleneck_nn_targets = bottleneck_nn_targets_full

    # Combined dataset (sampled)
    combined_nn_data = np.concatenate([corridor_nn_data, bottleneck_nn_data])
    combined_nn_targets = np.concatenate(
        [corridor_nn_targets, bottleneck_nn_targets]
    )
    if len(combined_nn_data) > sample_size:
        np.random.seed(44)  # New seed for combined sampling
        combined_indices = np.random.choice(
            len(combined_nn_data), sample_size, replace=False
        )
        combined_nn_data = combined_nn_data[combined_indices]
        combined_nn_targets = combined_nn_targets[combined_indices]

    # Print dataset info
    print("=== Neural Network Model Cross-Dataset Evaluation")
    print(
        f"Original dataset sizes: Corridor={len(corridor_nn_data_full)}, Bottleneck={len(bottleneck_nn_data_full)}"
    )
    print(
        f"Sampled dataset sizes: Corridor={len(corridor_nn_data)}, Bottleneck={len(bottleneck_nn_data)}"
    )
    print(f"Combined sampled size: {len(combined_nn_data)}")
    print(
        f"Speed optimization: Using max {sample_size} samples per dataset"
    )

    # Evaluation scenarios
    nn_evaluation_scenarios = [
        (
            "Corridor",
            "Corridor",
            corridor_nn_data,
            corridor_nn_targets,
            corridor_nn_data,
            corridor_nn_targets,
        ),
        (
            "Bottleneck",
            "Bottleneck",
            bottleneck_nn_data,
            bottleneck_nn_targets,
            bottleneck_nn_data,
            bottleneck_nn_targets,
        ),
        (
            "Corridor",
            "Bottleneck",
            corridor_nn_data,
            corridor_nn_targets,
            bottleneck_nn_data,
            bottleneck_nn_targets,
        ),
        (
            "Bottleneck",
            "Corridor",
            bottleneck_nn_data,
            bottleneck_nn_targets,
            corridor_nn_data,
            corridor_nn_targets,
        ),
        (
            "Combined",
            "Corridor",
            combined_nn_data,
            combined_nn_targets,
            corridor_nn_data,
            corridor_nn_targets,
        ),
        (
            "Combined",
            "Bottleneck",
            combined_nn_data,
            combined_nn_targets,
            bottleneck_nn_data,
            bottleneck_nn_targets,
        ),
        (
            "Combined",
            "Combined",
            combined_nn_data,
            combined_nn_targets,
            combined_nn_data,
            combined_nn_targets,
        ),
    ]

    nn_results = []

    print("\n" + "=" * 80)
    print(
        "NEURAL NETWORK MODEL EVALUATION RESULTS (TESTING - SMALL SUBSET)"
    )
    print("=" * 80)

    for (
        train_name,
        test_name,
        X_train,
        y_train,
        X_test,
        y_test,
    ) in nn_evaluation_scenarios:
        print(f"\nTrain: {train_name} | Test: {test_name}")
        print("-" * 50)

        try:
            if train_name == test_name:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train, y_train, test_size=0.5, random_state=42
                )
            else:
                np.random.seed(42)  # Optional: set seed for reproducibility

                # Subsample disjoint train/test sets from X_train and X_test sources
                # Ensures that the train and test sets are the same size for all scenarios
                train_indices = np.random.choice(
                    len(X_train),
                    min(len(X_train), sample_size // 2),
                    replace=False,
                )
                test_indices = np.random.choice(
                    len(X_test),
                    min(len(X_test), sample_size // 2),
                    replace=False,
                )

                X_train = X_train[train_indices]
                y_train = y_train[train_indices]
                X_test = X_test[test_indices]
                y_test = y_test[test_indices]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            print(
                f"Training set size: {len(X_train_scaled)}, Test set size: {len(X_test_scaled)}"
            )

            train_mse, train_std_mse, test_mse, test_std_mse = (
                train_with_bootstrap_validation(
                    X_train_scaled,
                    y_train,
                    X_test_scaled,
                    y_test,
                    layers=layers,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    activation=activation,
                    n_bootstrap=n_bootstrap,
                    verbose=verbose,
                )
            )

            result = {
                "Train Dataset": train_name,
                "Test Dataset": test_name,
                "Mean MSE Train": train_mse,
                "Std MSE Train": train_std_mse,
                "Mean MSE Test": test_mse,
                "Std MSE Test": test_std_mse,
                "Train Size": len(X_train_scaled),
                "Test Size": len(X_test_scaled),
            }
            nn_results.append(result)

            print(f"  Mean Train MSE: {train_mse:.6f}")
            print(f"  Std Train MSE: {train_std_mse:.6f}")
            print(f"  Mean Test MSE: {test_mse:.6f}")
            print(f"  Std Test MSE: {test_std_mse:.6f}")

        except Exception as e:
            print(f"Error training neural network: {e}")

    print("\n" + "=" * 80)
    print("NEURAL NETWORK SUMMARY TABLE")
    print("=" * 80)

    nn_results_df = pd.DataFrame(nn_results)
    print(
        nn_results_df[
            [
                "Train Dataset",
                "Test Dataset",
                "Mean MSE Train",
                "Std MSE Train",
                "Mean MSE Test",
                "Std MSE Test",
                "Train Size",
                "Test Size",
            ]
        ].to_string(index=False)
    )

    return nn_results_df
