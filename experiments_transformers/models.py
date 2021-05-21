import tensorflow as tf
from tensorflow_addons.layers import ESN
from tcn import TCN
from transformerDecoder import TransformerDecoderModel
from transformerDecoder2 import TransformerDecoderModel2
from transformerEncoderModel import TransformerEncoderModel



def mlp(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    hidden_layers=[32, 16, 8],
    dropout=0.0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Flatten()(inputs)  # Convert the 2d input in a 1d array
    for hidden_units in hidden_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def ernn(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    recurrent_units=[50],
    recurrent_dropout=0,
    return_sequences=False,
    dense_layers=[],
    dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.SimpleRNN(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.SimpleRNN(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def esn(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    recurrent_units=[50],
    return_sequences=False,
    dense_layers=[32],
    dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = ESN(recurrent_units[0], return_sequences=return_sequences_tmp, use_norm2=True)(
        inputs
    )
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = ESN(u, return_sequences=return_sequences_tmp, use_norm2=True)(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def lstm(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    recurrent_units=[50],
    recurrent_dropout=0,
    return_sequences=False,
    dense_layers=[],
    dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.LSTM(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.LSTM(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def gru(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    recurrent_units=[50],
    recurrent_dropout=0,
    return_sequences=False,
    dense_layers=[],
    dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.GRU(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.GRU(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def cnn(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    conv_layers=[64, 128],
    kernel_sizes=[7, 5],
    pool_sizes=[2, 2],
    dense_layers=[],
    dense_dropout=0.0,
):
    assert len(conv_layers) == len(kernel_sizes)
    assert 0 <= dense_dropout <= 1
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # First conv block
    x = tf.keras.layers.Conv1D(
        conv_layers[0], kernel_sizes[0], activation="relu", padding="same"
    )(inputs)
    if pool_sizes[0] and x.shape[-2] // pool_sizes[0] > 1:
        x = tf.keras.layers.MaxPool1D(pool_size=pool_sizes[0])(x)
    # Rest of the conv blocks
    for chanels, kernel_size, pool_size in zip(
        conv_layers[1:], kernel_sizes[1:], pool_sizes[1:]
    ):
        x = tf.keras.layers.Conv1D(
            chanels, kernel_size, activation="relu", padding="same"
        )(x)
        if pool_size and x.shape[-2] // pool_size > 1:
            x = tf.keras.layers.MaxPool1D(pool_size=pool_size)(x)
    # Dense block
    x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def tcn(
    input_shape,
    output_size=1,
    optimizer="adam",
    loss="mae",
    nb_filters=64,
    kernel_size=2,
    nb_stacks=1,
    dilations=[1, 2, 4, 8, 16],
    tcn_dropout=0.0,
    return_sequences=True,
    activation="linear",
    padding="causal",
    use_skip_connections=True,
    use_batch_norm=False,
    dense_layers=[],
    dense_dropout=0.0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        use_skip_connections=use_skip_connections,
        dropout_rate=tcn_dropout,
        activation=activation,
        use_batch_norm=use_batch_norm,
        padding=padding,
    )(inputs)
    # Dense block
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_rnn(func):
    return lambda input_shape, output_size, optimizer, loss, **args: func(
        input_shape=input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss="mae",
        recurrent_units=[args["units"]] * args["layers"],
        return_sequences=args["return_sequence"],
    )


def create_cnn(input_shape, output_size, optimizer, loss, conv_blocks):
    conv_layers = [b[0] for b in conv_blocks]
    kernel_sizes = [b[1] for b in conv_blocks]
    pool_sizes = [b[2] for b in conv_blocks]
    return cnn(
        input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss=loss,
        conv_layers=conv_layers,
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
    )

def TransformerDecoder(
    input_shape,
    output_size,
    N=3,
    d_model=256,
    h=8):

    model = TransformerDecoderModel(input_shape[-2],output_size,input_shape[-1],d_model,h,N)

    return model

def TransformerDecoder2(
    input_shape,
    output_size,
    N=3,
    d_model=256,
    h=8):

    model = TransformerDecoderModel2(input_shape[-2],output_size,input_shape[-1],d_model,h,N)

    return model

def TransformerEncoder(
    input_shape,
    output_size,
    N=3,
    d_model=256,
    h=8):

    model = TransformerEncoderModel(input_shape[-2],output_size,input_shape[-1],d_model,h,N)

    return model

model_factory = {
    "mlp": mlp,
    "ernn": create_rnn(ernn),
    "lstm": create_rnn(lstm),
    "gru": create_rnn(gru),
    "esn": create_rnn(esn),
    "cnn": create_cnn,
    "tcn": tcn,
    "trD_AR": TransformerDecoder,
    "trD2_AR": TransformerDecoder2,
    "trE": TransformerEncoder
}


def create_model(model_name, input_shape, **args):
    assert model_name in model_factory.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_factory[model_name](input_shape, **args)
