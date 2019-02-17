import json
import keras
from keras import Sequential, Input, Model
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np
import os
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


# dataset = get_dataset(src_path=data_file_path)

DATASETS = [file_ for file_ in os.listdir("all_new_mids")]
ret_datsets = [get_dataset("all_new_mids/"+d) for d in DATASETS]
wrapped = [[i["x"] for i in e] for e in ret_datsets]
rest = []
for e in wrapped:
    rest.extend(e)

X = np.array(rest)
# base_shape = X.shape[1]
X = X.reshape((-1, X.shape[1]))


def get_auto_encoder(input_shape):
    # this is the size of our encoded representations
    encoding_dim = 4  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(input_shape,))
    # "encoded" is the encoded representation of the input
    en_inter = Dense(32, activation='tanh')(input_img)
    en_inter = Dense(16, activation='tanh')(en_inter)
    encoded = Dense(encoding_dim, activation='tanh')(en_inter)
    # "decoded" is the lossy reconstruction of the input
    intermediate = Dense(16, activation='tanh')(encoded)
    intermediate = Dense(32, activation='tanh')(intermediate)
    decoded = Dense(input_shape, activation='sigmoid')(intermediate)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder, encoder, None


model_1, encoder_1, decoder_1 = get_auto_encoder(input_shape=X.shape[1])

from sklearn.model_selection import train_test_split

# x_train, x_test, _, _ = train_test_split(X, np.zeros(shape=(X.shape[1],)), test_size=0.33, random_state=42)

model_1.fit(X, X,
            epochs=200,
            batch_size=50,
            shuffle=True,
            validation_split=0.4)

model_1.save("ae.m")
encoder_1.save("encoder_1.m")

# encoded_imgs = encoder_1.predict(X)



