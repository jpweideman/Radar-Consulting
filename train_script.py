
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D

def train_convlstm_model(
    data_path="Data/ZH_radar_dataset.npy",
    save_path="checkpoints/normalized_convlstm_mse_model.keras",
    train_split=0.6,
    seq_len=10,
    filters1=32,
    filters2=16,
    kernel_size=(3, 3),
    loss_fn='mse',
    optimizer='adam',
    epochs=1,
    batch_size=2,
    verbose=1
):
    """
    Trains a ConvLSTM-based model for radar reflectivity forecasting using a sequence-to-one setup.

    This function loads radar data from a `.npy` file (data_path), normalizes it, creates sequences for training,
    builds a ConvLSTM neural network, and trains the model. The model is saved to the specified path (save_path).

    Args:
        - data_path (str): Path to the `.npy` radar dataset with shape (T, C, H, W).
        - save_path (str): Path where the trained model will be saved.
        - train_split (float): Proportion of the data to use for training (0 < train_split < 1).
        - seq_len (int): Length of the input sequence (number of past timesteps).
        - filters1 (int): Number of filters in the first ConvLSTM2D layer.
        - filters2 (int): Number of filters in the second ConvLSTM2D layer.
        - kernel_size (tuple): Kernel size for all ConvLSTM2D and Conv2D layers.
        - loss_fn (str or callable): Loss function for training. Can be 'mse' or a custom loss function.
        - optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for training.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size used for training.
        - verbose (int): Verbosity level for training output (0, 1, or 2).

    Returns:
        None
    """

    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load and preprocess data
    data = np.load(data_path)
    data = np.transpose(data, (0, 2, 3, 1))
    data[data < 0] = 0
    ZH_MAX = np.max(data)
    data_norm = data / ZH_MAX

    # Create sequences
    X, y = [], []
    for i in range(len(data_norm) - seq_len):
        X.append(data_norm[i:i+seq_len])
        y.append(data_norm[i+seq_len])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Train-test split
    split = int(len(X) * train_split)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    #Build model
    model = Sequential([
        Input(shape=(seq_len, 360, 240, 14)),
        ConvLSTM2D(filters=filters1, kernel_size=kernel_size, padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=filters2, kernel_size=kernel_size, padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=14, kernel_size=kernel_size, activation='relu', padding='same')
    ])

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,  # Turn off default logging
    )

    if not save_path.endswith(".keras"):
        save_path += ".keras"
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return model

def weighted_mae_high_reflectivity(threshold=30.0, zh_max=70.0, high_weight=5.0):
    """Returns a custom MAE loss that upweights pixels where ZH > threshold (in dBZ)."""
    norm_thresh = threshold / zh_max

    def loss(y_true, y_pred):
        # Weight = high_weight if y_true > threshold else 1.0
        weights = tf.where(y_true > norm_thresh, high_weight, 1.0)

        abs_error = tf.abs(y_true - y_pred)
        weighted_error = abs_error * weights

        return tf.reduce_mean(weighted_error)

    return loss
 
