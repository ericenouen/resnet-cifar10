# ResNet version does include the skip connections between different layers
import tensorflow as tf

# Cifar-10 has 10 classes and n relates to the number of layers
num_classes = 10
n = 9

# Uses He uniform because it was used in the initial paper
initializer = "he_uniform"

# The identity block keeps the feature map size the same
def identity_block(x, layers):
    input_val = x
    x = tf.keras.layers.Conv2D(layers, [3,3], padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv2D(layers, [3,3], padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = x + input_val
    return x


# The conv block halves the feature map size
def conv_block(x, layers):
    input_val = x
    x = tf.keras.layers.Conv2D(layers, [3,3], (2,2), padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv2D(layers, [3,3], padding='same', kernel_initializer=initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

    input_val = tf.keras.layers.Conv2D(layers, [1,1], (2,2), padding='same', kernel_initializer=initializer)(input_val)
    input_val = tf.keras.layers.BatchNormalization()(input_val)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = x + input_val
    return x


# Model is built according to the https://arxiv.org/pdf/1512.03385.pdf Section 4.2

input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name = "input_images")

x = tf.keras.layers.Conv2D(16, [3,3], padding='same', kernel_initializer=initializer)(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

for i in range(n):
    x = identity_block(x, 16)

x = conv_block(x, 32)
for i in range(n-1):
    x = identity_block(x, 32)

x = conv_block(x, 64)
for i in range(n-1):
    x = identity_block(x, 64)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
out = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=input_layer, outputs=out)

print(model.summary())