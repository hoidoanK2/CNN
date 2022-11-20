import tensorflow as tf
import cv2
from keras.utils import np_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#  CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),

])

#  model.summary()

#  print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = X_train.astype("float")
X_train /= 255

#  print(y_train.shape)
y_train = np_utils.to_categorical(y_train, 10)

model.compile(optimizer="sgd", loss="mean_squared_error", metrics="accuracy")

model.fit(X_train, y_train, epochs=15)

img_test = cv2.imread("anh3.png", 0)
#  cv2.imshow("anh test", img_test)
#  cv2.waitKey(0)
img_test = img_test.reshape(1, 28, 28, 1)
print(model.predict(img_test))
