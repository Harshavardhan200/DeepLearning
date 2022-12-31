import tensorflow as tf
from tensorflow import keras as ke
import matplotlib.pyplot as plt, numpy as np

(xtrain, ytrain), (xtest, ytest) = ke.datasets.mnist.load_data()
xtrain, xtest = xtrain / 255, xtest / 255
xtrain_flatten = xtrain.reshape(len(xtrain), 28 * 28)
xtest_flatten = xtest.reshape(len(xtest), 28 * 28)
print(xtest_flatten.shape)
# model = ke.Sequential([
#     ke.layers.Dense(10, input_shape=(784,), activation="sigmoid")
#
# ]) #model trained with only one input and output layer
# model = ke.Sequential([
#     ke.layers.Dense(100, input_shape=(784,), activation="relu"),
#     ke.layers.Dense(80, activation="relu"),
#     ke.layers.Dense(10, activation="sigmoid")
# ])   #model with two hidden layers
model = ke.Sequential([
    ke.layers.Flatten(input_shape=(28, 28)),
    ke.layers.Dense(100, activation="relu"),
    ke.layers.Dense(80, activation="relu"),
    ke.layers.Dense(10, activation="sigmoid")
])  # flattened in the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(xtrain_flatten, ytrain, epochs=5) #model with already flattened
model.fit(xtrain, ytrain, epochs=5)
model_json = model.to_json()
y_predict = model.predict(xtrain)
print(np.argmax(y_predict[0]))
plt.matshow(xtrain[0])
plt.show()
with open("binary_classification.h5", 'w') as mo:
    mo.write(model_json)
    print("model loaded successfully")
