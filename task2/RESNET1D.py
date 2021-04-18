from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, RNN, StackedRNNCells, \
    LSTMCell, Conv1D, Activation, MaxPooling1D, Add, Lambda, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input, Model
from tensorflow import keras


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def _bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)
    if dropout > 0:
        layer = Dropout(params["conv_dropout"])(layer)
    return layer


def add_conv_weight(layer, filter_length, num_filters, subsample_length=1, **params):
    layer = Conv1D(filters=num_filters, kernel_size=filter_length, strides=subsample_length, padding='same',
                   kernel_initializer=params["conv_init"])(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(layer, params["conv_filter_length"], params["conv_num_filters_start"],
                                subsample_length=subsample_length, **params)
        layer = _bn_relu(layer, **params)
    return layer


def resnet_block(layer, num_filters, subsample_length, block_index, **params):
    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(layer, dropout=params["conv_dropout"] if i > 0 else 0, **params)
        layer = add_conv_weight(layer, params["conv_filter_length"], num_filters, subsample_length if i == 0 else 1,
                                **params)
    layer = Add()([shortcut, layer])
    return layer


def get_num_filters_at_index(index, num_start_filters, **params):
    return 2 ** int(index / params["conv_increase_channels_at"]) * num_start_filters


def add_resnet_layers(layer, **params):
    layer = add_conv_weight(layer, params["conv_filter_length"], params["conv_num_filters_start"], subsample_length=1,
                            **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(layer, num_filters, subsample_length, index, **params)
    layer = _bn_relu(layer, **params)
    return layer


def add_output_layer(layer, **params):
    lstm_layer = RNN(StackedRNNCells([LSTMCell(params.get('lstm_units', 48), dropout=0.5)
                                      for _ in range(params.get('lstm_cells', 2))]))
    lstm_layer = Bidirectional(lstm_layer)(layer)
    shortcut = Dense(16)(layer)
    shortcut = Flatten()(shortcut)
    layer = Add()([shortcut, lstm_layer])
    layer = Dropout(0.5)(layer)
    layer = Dense(params["num_categories"])(layer)
    return Activation('sigmoid' if params["task_type"] is "classification" else "linear")(layer)


def add_compile(model, **params):
    adam = Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))
    sgd = SGD(lr=params["learning_rate"], momentum=params.get("sgd_momentum", 0.99), decay=params.get("sgd_decay", 2e-4))
    rmsp = RMSprop(lr=params["learning_rate"], decay=params.get("sgd_decay", 2e-4))
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=adam, metrics=[r2_score])


def build_network(**params):
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')
    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)
    return model
