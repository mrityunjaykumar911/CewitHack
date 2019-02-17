import json
import keras
from keras import Sequential, Input, Model
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np

data_file_path = "data_v1.json"


def get_dataset(src_path):
    decode = None
    with open(src_path) as fp:
        decoded = json.load(fp)
    assert decoded
    # parse that decoded into x component
    datas = []
    for key, each in decoded.items():
        x_ = [each["note_status"],
              each["channel"],
              each["note"],
              each["velocity"],
              each["time"],
              1 if key[0] == "U" else 0,
              1 if key[0] == "R" else 0,
              1 if key[0] == "M" else 0]
        data = {"x": x_,
                "tag": key}
        datas.append(data)
    return datas


dataset = get_dataset(src_path=data_file_path)

X = np.array([each["x"] for each in dataset])
# base_shape = X.shape[1]
X = X.reshape((X.shape[1], X.shape[0]))


def get_auto_encoder(input_shape):
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder, decoder


model_1, encoder_1, decoder_1 = get_auto_encoder(input_shape=X.shape[0])

from sklearn.model_selection import train_test_split

x_train, x_test, _, _ = train_test_split(X, np.zeros(shape=(1, X.shape[1])), test_size=0.33, random_state=42)

model_1.fit(x_train, x_train,
            epochs=50,
            batch_size=256,
            shuffle=True,
            validation_data=(x_test, x_test))

encoded_imgs = encoder_1.predict(x_test)

