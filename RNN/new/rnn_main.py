import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard


def ToTable(ue):
    ue_list = ue.vehicle_id.astype('category')
    ue_list = ue_list.cat.categories
    ue_list = list(ue_list)
    return ue_list


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
                                         activation='tanh',
                                         kernel_initializer=tf.keras.initializers.glorot_normal()))
    models.add(tf.keras.layers.Dense(8, kernel_regularizer=regularizers.l2(l=0.001)))
    models.add(tf.keras.layers.Dense(1, name='Output-Layer'))
    models.compile(optimizer=tf.keras.optimizers.Nadam(clipnorm=1.0),
                   loss=tf.keras.losses.Huber())
    return models


def Attention_SimpleRNN():
    inputs = tf.keras.Input(shape=(2, 1))
    # Encoder
    encoder = tf.keras.layers.SimpleRNN(16, return_sequences=True,
                                        kernel_initializer=tf.keras.initializers.glorot_normal())(inputs)
    # Attention mechanism
    attention = tf.keras.layers.Attention()([encoder, encoder])
    batch_norm = tf.keras.layers.BatchNormalization()(attention)
    residual = tf.keras.layers.Add()([attention, batch_norm])

    # Decoder
    decoder = tf.keras.layers.Dense(4)(residual)
    output = tf.keras.layers.Dense(1, activation="linear")(decoder)

    models = tf.keras.Model(inputs=inputs, outputs=output, name="RNN-Model")
    models.compile(optimizer=tf.keras.optimizers.Nadam(clipnorm=1.0),
                   loss=tf.keras.losses.huber)

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
    return models


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


if __name__ == '__main__':
    # read UEs data
    veh = pd.read_parquet('veh.parquet', engine='pyarrow')
    ped = pd.read_parquet('ped.parquet', engine='pyarrow')
    # query vehicle_list
    try:
        vehicle_list = list(pd.read_parquet('veh_list.parquet', engine='pyarrow')['veh_id'])
        pedestrian_list = list(pd.read_parquet('ped_list.parquet', engine='pyarrow')['ped_id'])

    except FileNotFoundError:
        vehicle_list = veh['vehicle_id'].astype('category')
        vehicle_list = list(vehicle_list.cat.categories)
        vehicle_list = pd.DataFrame(vehicle_list, columns=['vehicle_id'])
        vehicle_list.to_parquet('veh_list.parquet', engine='pyarrow')

        pedestrian_list = ped['ped_id'].astype('category')
        pedestrian_list = list(pedestrian_list.cat.categories)
        pedestrian_list = pd.DataFrame(pedestrian_list, columns=['ped_id'])
        pedestrian_list.to_parquet('ped_list.parquet', engine='pyarrow')
    veh_list = list(vehicle_list['vehicle_id'])
    ped_list = list(pedestrian_list['ped_id'])

    '''gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)'''
    # model initial
    model = Attention_SimpleRNN()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, mode='min', verbose=1)
    log_dir = 'path/to/save/logs'
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    # train
    for index, veh_id in enumerate(veh_list):
        temp = veh[veh['vehicle_id'] == veh_id]
        lens = len(temp)
        if lens <= 100:
            continue

        trainX = temp[['vehicle_x', 'vehicle_y']].values[:-1] / 1000.0
        trainY = temp[['vehicle_x', 'vehicle_y']].values[1:] / 1000.0

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

        model.fit(trainX,  # input data
                  trainY,  # target data
                  batch_size=64,
                  epochs=200,
                  validation_split=0.3,
                  callbacks=[early_stop],
                  use_multiprocessing=True)

        model.save('RNN.h5')

    for index, ped_id in enumerate(ped_list):
        temp = ped[ped['ped_id'] == ped_id]
        lens = len(temp)
        if lens <= 100:
            continue

        trainX = temp[['ped_x', 'ped_y']].values[:-1] / 1000.0
        trainY = temp[['ped_x', 'ped_y']].values[1:] / 1000.0

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

        model.fit(trainX,  # input data
                  trainY,  # target data
                  batch_size=64,
                  epochs=200,
                  validation_split=0.3,
                  callbacks=[early_stop],
                  use_multiprocessing=True)
        model.save('RNN.h5')

    print('Finish/')
