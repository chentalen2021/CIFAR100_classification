#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, layers, Sequential
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, GlobalMaxPooling2D,\
            Flatten, BatchNormalization, Dense, GlobalAveragePooling2D, Dropout

#Allow the GPU memory growth for deep learning
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
physical_devices
# %%
#Define an advanced res block
class Advanced_resblock(Layer):
    def __init__(self, n_filers, stride=1):
        super(Advanced_resblock, self).__init__()

        #Channel 1 -> conv2D + dilated-conv2D + MaxPool
            #1.1
        self.conv = Conv2D(n_filers, (3,3), strides=stride, padding="same")
        self.bn1 = BatchNormalization()
        self.relu = layers.LeakyReLU(alpha=0.3)
            #1.2
        if stride == 1:
            self.dconv = Conv2D(n_filers, (3,3), dilation_rate=2, padding="same")
        else:
            self.dconv = Sequential()
            self.dconv.add(Conv2D(n_filers, (3,3), padding="same", dilation_rate=2))
            self.dconv.add(MaxPool2D((2,2)))
        self.bn2 = BatchNormalization()
            #1.3
        self.maxpool = Sequential()
        self.maxpool.add(MaxPool2D(pool_size=(3,3), strides=stride ,padding="same"))
        self.maxpool.add(Conv2D(n_filers,(1,1)))
    
        #Channel 2
        if stride == 1:
            self.shortcut = lambda x:Conv2D(n_filers, (1,1), strides=1)(x)
        else:
            self.shortcut = Sequential()
            self.shortcut.add(Conv2D(n_filers, (1,1), strides=2))

    def call(self, inputs, training=None):
        #Conv2D
        x1 = self.conv(inputs)
        x1 = self.bn1(x1)
        #Dilated-conv2D
        x2 = self.dconv(inputs)
        x2 = self.bn2(x2)
        #MaxPool
        x3 = self.maxpool(inputs)

        # output from channel 1
        out_c1 = layers.Concatenate(axis=-1)([x1,x2,x3])

        #output form channel 2
        out_c2 = self.shortcut(inputs)
        out_c2 = layers.Concatenate(axis=-1)([out_c2, out_c2, out_c2])

        #Finalise the output
        out = layers.add([out_c1, out_c2])
        out = self.relu(out)

        return out

#%%
#Define res net
class ResNet(Model):
    def __init__(self, block_distribution, n_classes):  #[2,2,2,2]
        super(ResNet, self).__init__()
        #The first layer
        self.stem = Sequential([Conv2D(64, (3,3), strides=1),
                                BatchNormalization(),
                                layers.LeakyReLU(0.3),
                                MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same")])

        self.layer1 = self.build_resblock(32, block_distribution[0])
        self.layer2 = self.build_resblock(32, block_distribution[1], stride=2)
        self.layer3 = self.build_resblock(32, block_distribution[2], stride=2)
        self.layer4 = self.build_resblock(32, block_distribution[3], stride=2)

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
        res_blocks.add(Advanced_resblock(n_filters))

        for _ in range(1, n_blocks):
            res_blocks.add(Advanced_resblock(n_filters))
        
        return res_blocks

# %%
# Pre-define some ResNet models
def ResNet_advanced10(n_classes):
    print("Initiate ResNet_advanced10 successfully!")
    return ResNet([2,2,0,0], n_classes)

def ResNet_advanced18(n_classes):
    print("Initiate ResNet_advanced18 successfully!")
    return ResNet([2,2,2,2], n_classes)

def ResNet_advanced34(n_classes):
    print("Initiate ResNet_advanced34 successfully!")
    return ResNet([3,4,6,3], n_classes)

# %%
if __name__=="__main__":
    resnet_advanced18 = ResNet_advanced10(100)
    resnet_advanced18.build(input_shape=(None, 32, 32, 3))
    resnet_advanced18.summary()
# %%
inp = keras.Input(shape=(32,32,3))
x = MaxPool2D((3,3), strides=1, padding="same")(inp)
x
# %%
