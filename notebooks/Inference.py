
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * x + self.beta

generator_g = load_model(
    "generator_g_epoch_200.keras",
    custom_objects={
        "InstanceNormalization": InstanceNormalization(),
        "ReflectionPadding2D": ReflectionPadding2D()
    }
)

generator_f = load_model(
    "generator_f_epoch_200.keras",
    custom_objects={
        "InstanceNormalization": InstanceNormalization(),
        "ReflectionPadding2D": ReflectionPadding2D
    }
)


def preprocess_image(path, size=128):

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, (size, size))

    img = (tf.cast(img, tf.float32) / 127.5) - 1

    img = tf.expand_dims(img, 0)

    return img


def postprocess(img):

    img = (img + 1) * 127.5
    img = tf.clip_by_value(img, 0, 255)

    img = sharpen(img)
    return tf.cast(img, tf.uint8)


def sharpen(img):
   gaussian = cv2.GaussianBlur(img, (0,0), 1.2)
   sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
   img= sharpened

   #  highpass = img - gaussian
   #  img = img + highpass*0.5 
   #  img = np.clip(img, 0, 255).astype(np.uint8)

   return img



def young_to_old(path):

    img = preprocess_image(path)

    generated = generator_g(img, training=False)

    return postprocess(generated[0])


def old_to_young(path):

    img = preprocess_image(path)

    generated = generator_f(img, training=False)

    return postprocess(generated[0])