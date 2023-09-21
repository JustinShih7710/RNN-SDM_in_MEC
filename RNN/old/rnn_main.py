import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from keras.layers import Dense, SimpleRNN
from matplotlib import pyplot as plt


def ToTable(df1):
    veh_list = df1.vehicle_id.astype('category')
    veh_list = veh_list.cat.categories
    veh_list = pd.DataFrame(veh_list.to_numpy(), columns=['vehicle_id'])
    return veh_list


def Simple_RNN():
    models = tf.keras.Sequential(name="RNN-Model")  # Model
    models.add(tf.keras.layers.SimpleRNN(64, input_shape=(2, 1), name='Hidden-Recurrent-Layer', return_sequences=True,
                                         recurrent_dropout=0.3))
    models.add(
        tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_regularizer=regularizers.l2(l=0.001)))
    models.add(tf.keras.layers.Dropout(0.25))
    models.add(tf.keras.layers.Dense(1, name='Output-Layer'))

    models.compile(optimizer=tf.keras.optimizers.Nadam(),
                   loss=tf.keras.losses.huber)
    return models


def Xavier_RNN():
    models = tf.keras.Sequential(name="RNN-Model")  # Model
    models.add(tf.keras.layers.SimpleRNN(16, input_shape=(2, 1), name='Hidden-Recurrent-Layer', return_sequences=True,
                                         kernel_initializer=tf.keras.initializers.glorot_normal()))
    models.add(tf.keras.layers.Dense(8, kernel_regularizer=regularizers.l2(l=0.001)))
    models.add(tf.keras.layers.Dense(1, name='Output-Layer'))
    models.compile(optimizer=tf.keras.optimizers.Nadam(clipnorm=1.0, clipvalue=0.25),
                   loss=tf.keras.losses.Huber(delta=1.5))
    return models


def Attention_SimpleRNN():
    inputs = tf.keras.Input(shape=(2, 1))
    # Encoder
    encoder = tf.keras.layers.SimpleRNN(64, return_sequences=True,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                        activation='tanh')(inputs)
    batch_norm = tf.keras.layers.BatchNormalization()(encoder)
    activation = tf.keras.activations.sigmoid(batch_norm)
    # Attention mechanism
    attention = tf.keras.layers.Attention()([activation, activation])
    # Decoder
    decoder = tf.keras.layers.Dense(16)(attention)
    output = tf.keras.layers.Dense(1, activation="linear")(decoder)

    models = tf.keras.Model(inputs=inputs, outputs=output, name="RNN-Model")
    models.compile(optimizer=tf.keras.optimizers.Nadam(clipnorm=1.0, clipvalue=0.25, learning_rate=0.01),
                   loss=tf.keras.losses.Huber(delta=1.5))

    return models


def C_RNN():
    models = tf.keras.Sequential(name="CNN+RNN-Model")
    models.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(2, 1), padding='same',
                                      name='Conv1D-Layer'))
    models.add(tf.keras.layers.SimpleRNN(64, activation=tf.keras.activations.tanh, return_sequences=True, dropout=0.3,
                                         recurrent_dropout=0.3, name='RNN-Layer'))
    models.add(
        tf.keras.layers.Dense(16, activation=tf.keras.activations.selu, kernel_regularizer=regularizers.l2(l=0.001),
                              name='Dense-Layer'))
    models.add(tf.keras.layers.Dropout(0.25, name='Dropout-Layer'))
    models.add(tf.keras.layers.Dense(1, activation='linear', name='Output-Layer'))

    models.compile(optimizer=tf.keras.optimizers.Nadam(clipnorm=1.0),
                   loss=tf.keras.losses.huber)
    return model


def show_detail(losses: dict, val_losses: dict, index: int):
    epochs = 100
    loss = np.zeros((epochs, 1), dtype=np.float32)
    val_loss = np.zeros((epochs, 1), dtype=np.float32)
    for i in range(0, index):
        for n in range(0, len(losses[i])):
            loss[n] += losses[i][n]
        for n in range(0, len(val_losses[i])):
            val_loss[n] += val_losses[i][n]
    # loss = loss / index
    # val_loss = val_loss / index
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training loss & Validation loss')
    plt.legend()
    plt.show()


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.losses = {}
        self.val_losses = {}
        self.index = 0

    def on_epoch_end(self, epoch, logs=None):
        self.losses[self.index].append(logs.get('loss'))
        self.val_losses[self.index].append(logs.get('val_loss'))

    def on_train_begin(self, logs=None):
        self.losses[self.index] = []
        self.val_losses[self.index] = []

    def on_train_end(self, logs=None):
        self.index += 1


if __name__ == '__main__':
    dtypes = {'data_timestep': int, 'vehicle_id': str, 'vehicle_x': float, 'vehicle_y': float}
    # read veh.csv
    df = pd.read_csv('veh.csv', dtype=dtypes)
    try:
        # query vehicle_list
        vehicle_list = pd.read_csv('veh_list.csv')
        # cal velocity
        df2 = pd.read_csv('veh_vector.csv')
    except FileExistsError:
        vehicle_list = ToTable(df)
        dict1 = []
        for vehicle_id in vehicle_list.vehicle_id:
            time_interval = 1
            vehicle_x = np.array(df[df['vehicle_id'] == vehicle_id]['vehicle_x'])[0:10]
            vehicle_y = np.array(df[df['vehicle_id'] == vehicle_id]['vehicle_y'])[0:10]
            distance = np.sqrt(np.diff(vehicle_x) ** 2 + np.diff(vehicle_y) ** 2)
            speed = distance / time_interval
            d = {'vehicle_id': vehicle_id, 'vector': np.average(speed)}
            dict1.append(d)
        df2 = pd.DataFrame(dict1)

    ue_list = df['vehicle_id'].astype('category')
    ue_list = list(ue_list.cat.categories)[0:50]

    lens = []
    for veh_id in ue_list:
        lens.append(len(df[df['vehicle_id'] == veh_id]))
    lens = np.array(lens)
    tmp = []
    for index, veh_id in enumerate(ue_list):
        if (lens > lens.mean())[index]:
            tmp.append(veh_id)
    ue_list = tmp
    del dtypes, vehicle_list, df2, tmp
    gc.collect()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # model initial
    model = Xavier_RNN()

    # callback
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
    checkpoint_path = 'path/to/save/model/weights.h5'
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=0,
        save_weights_only=True
    )
    log_dir = 'path/to/save/logs'
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    loss_history = LossHistory()
    for index, veh_id in enumerate(ue_list):
        temp = df[df['vehicle_id'] == veh_id]

        trainX = temp[['vehicle_x', 'vehicle_y']].values[:-1] / 1000.0
        trainY = temp[['vehicle_x', 'vehicle_y']].values[1:] / 1000.0

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

        model.fit(trainX,  # input data
                  trainY,  # target data
                  batch_size=16,
                  epochs=100,
                  validation_split=0.3,
                  callbacks=[early_stop, loss_history],
                  use_multiprocessing=True)
    print('Finish/')

