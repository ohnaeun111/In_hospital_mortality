import tensorflow as tf
from tensorflow.keras import Model, optimizers

def define_model(input_size=149):
    x_input = tf.keras.Input(shape=(input_size,), name='input_data')

    x = tf.keras.layers.Dense(units=128)(x_input)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(units=64)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(units=32)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(units=16)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(units=8)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(units=4, kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x_out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=x_input, outputs=x_out)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.003), loss='binary_crossentropy', metrics=['accuracy'])
    return model
