#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, layers, Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D,\
            Flatten, BatchNormalization, Dense, GlobalAveragePooling2D, Dropout

#Allow the GPU memory growth for deep learning
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# %%
#Build the residual block
class BasicBlock(Layer):
    def __init__(self, n_filters, stride=1):
        super(BasicBlock, self).__init__()

        #Channel 1 ---- 2 convolution layers
        self.conv1 = Conv2D(n_filters, (3,3), strides=stride, padding="same",\
                            kernel_regularizer=keras.regularizers.l2(4e-3))
        self.bn1 = BatchNormalization()
        self.relu = layers.Activation("relu")

        self.conv2 = Conv2D(n_filters, (3,3), strides=1, padding="same",\
                            kernel_regularizer=keras.regularizers.l2(4e-3))
        self.bn2 = BatchNormalization()

        #Channel 2 ---- shortcut
        if stride != 1:
            self.shortcut = Sequential()
            self.shortcut.add(Conv2D(n_filters, (1,1), strides=2))
        else:
            self.shortcut = lambda x:x
        
    def call(self, inputs, training=None):
        #[b, h, w, c]
        out = self.conv1(inputs)
        out = Dropout(0.3)(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = Dropout(0.3)(out)
        out = self.bn2(out)
        out = self.relu(out)

        identity = self.shortcut(inputs)

        output = layers.add([out, identity])
        output = self.relu(output)

        return output
#%%
inp = keras.Input(shape=(32,32,3))
layer1 = BasicBlock(64, stride=2)(inp)
layer1

# %%
# Build ResNet
class ResNet(Model):
    def __init__(self, block_distribution, n_classes):  #[2,2,2,2]
        super(ResNet, self).__init__()
        #The first layer
        self.stem = Sequential([Conv2D(64, (3,3), strides=1),
                                BatchNormalization(),
                                layers.Activation("relu"),
                                MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same")])

        self.layer1 = self.build_resblock(64, block_distribution[0])
        self.layer2 = self.build_resblock(128, block_distribution[1], stride=2)
        self.layer3 = self.build_resblock(256, block_distribution[2], stride=2)
        self.layer4 = self.build_resblock(512, block_distribution[3], stride=2)

        #output: [b, 512, h, w]
            #Unknown the specific h and w, so global average it -> [b, 512, 1, 1]
        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(n_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = Flatten()(self.avgpool(x))
        output = self.fc(x)

        return output
    
    def build_resblock(self, n_filters, n_blocks, stride=1):

        res_blocks = Sequential()
        res_blocks.add(BasicBlock(n_filters, stride))

        for _ in range(1, n_blocks):
            res_blocks.add(BasicBlock(n_filters, stride=1))
        
        return res_blocks
# %%
# Pre-define some ResNet models
def ResNet18(n_classes):
    return ResNet([2,2,2,2], n_classes)

def ResNet34(n_classes):
    return ResNet([3,4,6,3], n_classes)


#%%
if __name__=="__main__":
    resnet18 = ResNet18(100)
    resnet18.build(input_shape=(None, 32, 32, 3))
    resnet18.summary()
# %%
