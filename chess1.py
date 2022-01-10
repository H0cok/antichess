import chess
import chess.engine
import random
import numpy
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks


from tensorflow.keras.callbacks import ModelCheckpoint







squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    # this is the 3d matrix
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

    # here we add the pieces's view on the matrix
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3,
                          padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    return models.Model(inputs=board3d, outputs=x)


def build_model_residual(conv_size, conv_depth):
    board3d = layers.Input(shape=(14, 8, 8))
    # adding the convolutional layers
    x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(board3d)
    for _ in range(conv_depth):
        previous = x
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, previous])
        x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, 'sigmoid')(x)

    return models.Model(inputs=board3d, outputs=x)


def get_dataset():
    container = numpy.load('go0_1000deep.npz')
    b, v = container['b'], container['v']
    print(max(v), min(v))
    v = numpy.asarray(v / abs(v).max() / 2 + 0.5, dtype=numpy.float32)
    return b, v


x_train, y_train = get_dataset()
x_train.transpose()
print(x_train.shape)
print(y_train.shape)
model = build_model_residual(32, 4)

model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
model.summary()
checkpoint_filepath = '/tmp/checkpoint/'
model_checkpointing_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
)
model.fit(x_train, y_train,
          batch_size=2048,
          epochs=1000,
          verbose=1,
          validation_split=0.1,
          callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                     callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4),
                     model_checkpointing_callback])

print(5)
model.save('model0_1000_32_4res.h5')
