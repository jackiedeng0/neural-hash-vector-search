import os

os.environ["KERAS_BACKEND"] = "jax"

import keras

# embedding shape
input_shape = (50,)
# output shape
hash_bits = 8

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(hash_bits, keras.activations.sigmoid)
    ]
)

model.summary()

model.compile(
    loss=keras.losses.MeanAbsoluteError(),
    optimizer=keras.optimizers.Adam(),
)
