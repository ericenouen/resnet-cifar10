from cifar10 import Cifar
import tensorflow as tf
import numpy as np
import model_cifar10
import helper
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

import gc

'''
Edit which class you import in order to change from ResNet to Regular
model_cifar10- ResNet
mode_cifar10plain- Regular

Edit the filename variable in order to change where it saves to
Either /resnet/ or /plain/
3n/5n/7n/9n

Edit inside of model_cifar10.py or model_cifar10plain.py which
n value you want to specify.
'''

batch_size = 128
learning_rate = 0.1
no_of_epochs = 350

filename = "data/resnet/9n"
# (Number)n represents the n value of the model, n=3 means 20 layers


cifar = Cifar(batch_size=batch_size)

# Load training/testing data from a file
train = cifar.return_train()
inp, out = helper.transform_to_input_output(train, dim=model_cifar10.num_classes)
test = cifar.return_test()
inp_test, out_test = helper.transform_to_input_output(test, dim=model_cifar10.num_classes)


# Apply transformations from paper
datagen = ImageDataGenerator(
    featurewise_center=True,
    width_shift_range=0.125, # Shift left/right 4 pixels
    height_shift_range=0.125, # Shift up/down 4 pixels
    horizontal_flip=True,
    validation_split=0.1)
testdatagen = ImageDataGenerator(featurewise_center=True)

# Fit generators to input and create a training, cross-validation, and testing generator
datagen.fit(inp)
testdatagen.fit(inp)
train_generator = datagen.flow(inp, out, batch_size = batch_size, subset='training')
val_generator = datagen.flow(inp, out, batch_size = batch_size, subset='validation')
test_generator = testdatagen.flow(inp_test, out_test, batch_size=batch_size)

acc_list = []
acc_list_test = []

resnet = model_cifar10.model

# Did not apply the weight decay because it was not clear how often it was applied.

# Set up model and optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
resnet.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

for epoch in range(no_of_epochs):
    # Fit model for each epoch
    print(str(epoch+1) + "/" + str(no_of_epochs) + " ")
    history = resnet.fit_generator(train_generator, verbose=1, validation_data=val_generator, shuffle=True)

    # Reduce learning rate at 150 epochs and 250 epochs
    if (epoch == 150):
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate/10, momentum=0.9)
        resnet.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    if (epoch == 250):
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate/100, momentum=0.9)
        resnet.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    merge = tf.summary.merge_all()
    acc_list.append(history.history['acc'])
    
    # Evaluate test dataset
    testloss, testacc = resnet.evaluate_generator(test_generator, verbose=1)
    acc_list_test.append(testacc)

    
    gc.collect()


# Saves the train and test accuracy to their respective files
with open(filename + "_train", "w") as f:
    for acc in acc_list:
        f.write(str(acc[0]) +"\n")

with open(filename + "_test", "w") as f:
    for acc in acc_list_test:
        f.write(str(acc) +"\n")
