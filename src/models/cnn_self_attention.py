from tensorflow.keras import layers, models

class SelfAttention(layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=True)

    def call(self, inputs):
        q = layers.dot([inputs, self.W])
        a = layers.dot([inputs, self.W])
        attention = layers.dot([q, a], axes=[2, 1])
        attention = layers.Activation('softmax')(attention)
        output = layers.dot([attention, inputs])
        return output

def build_cnn_self_attention(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = SelfAttention()(x)

    outputs = layers.Dense(8, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model