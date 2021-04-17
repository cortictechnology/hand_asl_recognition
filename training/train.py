import numpy as np # linear algebra
import cv2
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D , Flatten , Dropout , BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, Add
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import regularizers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from sklearn.preprocessing import LabelBinarizer

characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

label_binarizer = LabelBinarizer()
input_size = 224

num_train = 0
print("reading training data")
for i in range(len(characters)):
    char = characters[i]
    for img in os.listdir("./" + char):
        if img.find("jpeg") != -1 or img.find("jpg") != -1:
            num_train = num_train + 1

print(num_train)

x_train = np.zeros((num_train, input_size, input_size, 3))
y_train = np.zeros((num_train, 1))

count = 0
for i in range(len(characters)):
    char = characters[i]
    for img in os.listdir("./" + char):
        if img.find("jpeg") != -1 or img.find("jpg") != -1:
            image = cv2.imread("./" + char + "/" + img)
            image = cv2.resize(image, (input_size, input_size))
            x_train[count] = image
            y_train[count] = i
            count += 1
            print(img, i)

idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[idx], y_train[idx]

y_train = label_binarizer.fit_transform(y_train)

#print(y_train)
#print(y_train.shape)

count = 0
print("reading test data")
num_test = 0

for i in range(len(characters)):
    char = characters[i]
    for img in os.listdir("./test/" + char):
        if img.find("jpeg") != -1 or img.find("jpg") != -1:
            num_test = num_test + 1

print(num_test)

x_test = np.zeros((num_test, input_size, input_size, 3))
y_test = np.zeros((num_test, 1))

for i in range(len(characters)):
    char = characters[i]
    for img in os.listdir("./test/" + char):
        if img.find("jpeg") != -1 or img.find("jpg") != -1:
            image = cv2.imread("./test/" + char + "/" + img)
            image = cv2.resize(image, (input_size, input_size))
            x_test[count] = image
            y_test[count] = i
            count += 1
            print(img, i)

idx = np.random.permutation(len(x_test))
x_test, y_test = x_test[idx], y_test[idx]

y_test = label_binarizer.fit_transform(y_test)

#print(y_test)
#print(y_test.shape)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

def unfreeze_model(model):
    # Unfreeze the entire model, can also choose to unfreeze only layers after the 100th layer
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

inputs = layers.Input(shape=(input_size,input_size, 3))
model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs,
    input_shape=(input_size,input_size,3)
)

# Freeze the base model
model.trainable = False

x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)
dropout_rate = 0.2
x = layers.Dropout(dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(24, activation="softmax", name="pred")(x)
model = tf.keras.Model(inputs, outputs, name="MobileNet")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

print(y_train.shape)
print(y_test.shape)

early_stopping_monitor = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=10,
    verbose=2,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 40 , validation_data = (x_test, y_test), callbacks=[early_stopping_monitor])

print("Accuracy of the model in first stage is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

unfreeze_model(model)
early_stopping_monitor = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=14,
    verbose=2,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

epochs = 70  # @param {type: "slider", min:8, max:50}
history = model.fit(datagen.flow(x_train,y_train, batch_size = 128), epochs=epochs, validation_data = (x_test, y_test), callbacks=[early_stopping_monitor])
print("Accuracy of the final model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

## Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"))
## Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
## Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)
