import tensorflow as tf


def q_network(state_size, action_size, seed, units=(64, 64)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),
        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.Dense(units[1], activation='relu'),
        tf.keras.layers.Dense(action_size),
    ])
    return model
