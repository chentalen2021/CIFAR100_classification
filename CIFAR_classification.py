#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, Input, Model, regularizers
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Activation,BatchNormalization, MaxPooling2D, Flatten
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
# import centre_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["tensorflow.keras_BACKEND"] = "plaidml.tensorflow.keras.backend"

#Allow the GPU memory growth for deep learning
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

physical_devices
#%%
#Display the device the current operation assigned to
# tf.debugging.set_log_device_placement(True)

# %%
#Load CIFAR100 dataset
batch_size = 128
    #image size = 32 * 32
    #data split: 60k -> 50k + 10k
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
x_train.shape, y_train.shape, x_test.shape, y_test.shape
# %%
#Squeeze the redundant 1-dimension in the y set
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

y_train.shape, y_test.shape
#%%
# y, idx, count = tf.unique_with_counts(y_train)
# y, count
# %%
#Define the preprocess method
def preprocess(x, y):
    mean = np.mean(x)
    std = np.std(x)

    x = (x-mean)/(std+1e-7)
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    
    return x, y

# %%
#Transform the dataset into Tensor dataset
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

#%%
#data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # Set input mean to 0 over the dataset, feature-wise.
    samplewise_center=False,  # Set each sample mean to 0.
    featurewise_std_normalization=False,  # Divide inputs by std of the dataset, feature-wise.
    samplewise_std_normalization=False,  # Divide each input by its std.
    zca_whitening=False,  # Apply ZCA whitening
    rotation_range=15,  # Degree range for random rotations
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

#The generation of the augmented images requires 
#transforming the y_train into a class vector (integers) to binary class matrix
y_train_vector = tf.keras.utils.to_categorical(y=y_train, num_classes=100)

db_train = datagen.flow(x_train, y_train_vector, batch_size=batch_size)

#%%
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
next(iter(db_train))[0].shape, next(iter(db_train))[1].shape

#%%
#Define summary variable
#current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_root = "D:\\Python\\Tensorflow\\logs"
path_train=log_root+"\\CIFAR100\\train"
path_test=log_root+"\\CIFAR100\\valid"

train_wr = tf.summary.create_file_writer(path_train)
valid_wr = tf.summary.create_file_writer(path_test)
#%%
# #Use 13-Conv layer model
# import Conv_13
# cnn = Conv_13.Conv13_model()

# #Use ResNet model
# import ResNet
# cnn = ResNet.ResNet18(n_classes=100)
# cnn = ResNet.ResNet34(n_classes=100)

#Use Advanced ResNet model
import ResNet_advanced
cnn = ResNet_advanced.ResNet_advanced18(n_classes=100)
# cnn = ResNet_advanced.ResNet_advanced18(n_classes=100)

cnn.build(input_shape=[None, 32,32,3])

# %%
#Train the cnn
    #Define an optimizer and acc metric
optimiser = tf.optimizers.Adam(learning_rate=1e-4)
acc = keras.metrics.SparseCategoricalAccuracy()

    #Training 30 epochs
for epoch in range(200):
    for step, (x, y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            logits = cnn(x)
            logits = tf.reshape(logits, [-1, 100])
            
            #compute loss
#             y_onehot = tf.one_hot(y, depth=100)

            loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
            loss_train = tf.reduce_mean(loss)
            
            #compute acc
            y_True = tf.math.argmax(y, axis=1, output_type=tf.dtypes.int32)
            acc.update_state(y_True, logits)

        grads = tape.gradient(loss_train, cnn.trainable_variables)

        optimiser.apply_gradients(zip(grads, cnn.trainable_variables))
        
        if step%200 == 0:
            print("epoch",epoch, "step", step, "loss", loss_train)
        
        #Set up a break condition for data augmentation, otherwise it will infinitely generate images
        if step > x_train.shape[0] // batch_size:
            break
        
    #Print the loss in trainng process for observation
    print("epoch",epoch, "loss: ", loss_train)
    
    #Write the summary of each epoch into the TensorBoard for real-time plotting of training process
    with train_wr.as_default():
        tf.summary.scalar("training&validation loss", float(loss_train), step=epoch)
        tf.summary.scalar("training&validation acc", float(acc.result()), step=epoch)
    
    #reset the acc between training and validating in each epoch
    acc.reset_states()
    
    #Observe the trained cnn's performance after each epoch on validation set
    for x_valid, y_valid in db_test:
        logits_valid = cnn(x_valid)
        logits_valid = tf.reshape(logits_valid, [-1, 100])
        
        y_valid_onehot = tf.one_hot(y_valid, depth=100)
        
        loss = tf.losses.categorical_crossentropy(y_valid_onehot, logits_valid, from_logits=True)
        loss_valid = tf.reduce_mean(loss)
        
        acc.update_state(y_valid, logits_valid)
    
    #Write the summary to the TensorBoard after each epoch
    with valid_wr.as_default():
        tf.summary.scalar("training&validation loss", float(loss_valid), step=epoch)
        tf.summary.scalar("training&validation acc", float(acc.result()), step=epoch)
    
    acc.reset_states()



# %%