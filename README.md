# CIFAR100_classification
CIFAR100 is a classical dataset for image classification practice. It contains 60000 images of the size 32*32*3, including 100 types of animal images. The dataset is pre-splitted into training set (50000 images) and testing set (10000 images). 

Initially, I developed a 13-layer Convolution model for classification, the loss and acc of training and testing (used as validation set) are shown in TensorBoard. 
The screen-shot of the training process is below:

![image](https://user-images.githubusercontent.com/80739689/113241364-0b64a280-930b-11eb-8601-0d04c3f1d116.png)

_Fig 1.1 Training process_

Overfitting could be seen from the chart. Then I used data augmentation strategy to generate more images to alleviate this situation.

![image](https://user-images.githubusercontent.com/80739689/113241611-87f78100-930b-11eb-8c03-ec5e1a0e6a1d.png)

_Fig 1.2 Training process with data augmentation_

Overfitting still exists, then I added Dropout layers to the model:

![image](https://user-images.githubusercontent.com/80739689/113241792-ddcc2900-930b-11eb-9cc9-28c7656fcb44.png)

_Fig 1.3 Training process with data augmentation and dropout strategy_

After that, I also used kernel regularisation strategy:

![image](https://user-images.githubusercontent.com/80739689/113241883-153ad580-930c-11eb-877e-1f179cf41a0e.png)

_Fig 1.4 _Training process with data augmentation, dropout strategy, and regularisation

During the above steps, different dropout and kernel regularisation rates/ positions are experimented and only the best one are presented.

The model seems hardly to be improved. I tried a shallower structure of the model to test if the low accuracy of validation set is still due to overfitting:

![image](https://user-images.githubusercontent.com/80739689/113242441-7adb9180-930d-11eb-9ee9-b53f7cf521f2.png)

_Fig 1.5 Training process of a shallower structure with the same strategies used

The validation accuracy is a little better, but the training process seems not so stable as the validation loss notably fluctuates. This means the low accuracy issue can not be well-solved by simply using shallower structure, or it is not so related to overfitting now.
Hence, I goes towards the opposite direction by developing the classifical ResNet18/ 34 from scratch and experimented their performance on the augmented dataset.

![ResNet18](https://user-images.githubusercontent.com/80739689/113242232-f4bf4b00-930c-11eb-97e0-0a56c8220ae6.PNG)
![ResNet34](https://user-images.githubusercontent.com/80739689/113242263-09034800-930d-11eb-81f2-b72dcd7a6a58.PNG)

_Fig 1.6 Training process of ResNet18(upper) and ResNet34(down)

The problem is still not solved. Both the shallower and deeper CNN architecture can not bring any notable improvement. Therefore, I guess the models used above lack of the ability to capture enough features of the images from different perspectives. The image quality is 32*32 which is so low even for human eyes to perceive. So, I consider to use a more complex model to extract different image features in parallel. Motivated by the GoogleNet/ Inception model, I decide to combine the concept of GoogleNet model and the ResNet to try to solve the problem. This means, in each Residual block, a convolution, a dilated convolution, and a max pooling layers is used to extract different features from image data, followed by a concatenation of these three features. A shortcut is also used to jump every 2 layers in one block. 

Furthermore, the learning rate is found important for optimising the DNN. Smith (2017) proposes a Cyclical Learning Rate (CLR) strategy for tuning the model. It avoids the fixed learning rate quandary ---- low learnnig rate leading to slow convergence while high lr for Gradient exploding. According to this paper, the following steps are used in my experiment.

One find the maximal and minimal lr range by plotting the accuracy and different lr values.


_Fig 1.7 Accuracy over different learnig rates

The ascending point for acc is 1e-5
