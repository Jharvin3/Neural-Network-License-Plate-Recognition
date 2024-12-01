# Neural-Network-License-Plate-Recognition
Introduction
For this project we will be working in a group of three people: Michelle Gore, Jamaine Harvin, and Alsion Tafa. We plan to use Tensorflow to create a neural network that reads out the numbers and letters from an image of a license plate. We will use a dataset of license plate images with labels from Kaggle to train and test our network.

Discussion
To start creating our network we needed to load and preprocess the data. Kaggle has a library that allows us to import our data into the code. Once we had the images we resized each image and converted each image to grayscale so the network does not have to deal with input of different sizes or read a color variable. We used the cv2 image library to do this.
We built our model with twenty layers. First, we use convolution layers, which extract the features of the image using a kernel function. It takes a tensor representation of the image, I=(height, width, channel) and gives us a feature map, denoted by F[i,j]=(I*K)[i,j], where K is the learned kernel that gets passed over the image tensor. In our case we converted our image to grayscale, so channel = 1. The ij-th entry of our resultant feature map is given by:

Where x and y represent the rows and columns of the kernel and i and j represent the rows and columns of the resultant feature map.
The feature map is then applied to the activation function ‘relu’: R(x)=max(0,x). It replaces all of the negative values in the feature map with zero, adding non-linearity to the model.
The batch normalization layers come after each convolution layer, they take the resultant feature map of the convolution and normalizes its outputs xi so that they have a mean,  of 0 and a variance 2 of 1. The normalized feature map is given by:

Where ϵ is a small constant added to the denominator to prevent division by zero.
Once the batch is normalized it is then scaled and shifted by learnable parameters λ and β. Giving us yi =xi+. These parameters allow the network to properly adjust the batch so its values are the most optimal for the training process.
The max pooling layers come after each batch normalization layer. They are used to reduce the dimensionality of each feature map. Max pooling divides the feature map into separate pooling regions and then outputs the maximum value of each region. In our network the size of these regions is a 2x2 grid.These values represent the most prominent features of the feature map like corners and edges.
After the convolution comes the flatten layer. The flatten layer takes our three dimensional feature map and turns it into a one dimensional vector. By turning the feature maps into a vector it reduces the amount of computations the network has to do while training. This is the last layer before the network starts the image classification.
The dense layer is where the network makes its guesses. It takes in the input from the flattened array and separates it into different nodes or neurons, zj. Where, 

and wij are the weights that connect the previous node i to the current node j, xi is the output of the previous node, and bj is the bias for the current node.
The network updates the values of the weights and biases to make a connection between the current input and the desired output. This allows the network to learn patterns and make a prediction based on the patterns it is learning. It then applies the activation function ‘relu’ to the output.
We use dropout, reshape, and softmax layers to assist the dense layer in making predictions. Dropout randomly sets half of the neuron weights to zero to prevent overfitting. Reshape reorganizes the data, there are 6 values on a license plate, each having 36 possible values. The softmax layer converts the neurons to a probability distribution, scaling the values of the input in between (0,1) to simplify the network's prediction process.


Before we can train the model we need to compile the model, which includes the following settings: Optimizer, Loss Function, and the Metrics. 

The Optimizer is updated based on the data it sees and its loss function. We used the ‘adam’ optimizer which involves a combination of two gradient descent methodologies:

Momentum:
 ⍵t+1=⍵t - ꭤmt 
where, 
mt=βmt + (1 - β) *[∂L/∂wt]


Root Mean Square Propagation (RSMP):

				⍵t+1=⍵t - ꭤt/(vt+ε)1/2*[∂L/∂wt]
				where,
				vt=βvt-1  +(1 - β) *[∂L/∂wt]2 

So, taking the two formulas from the above methods, we get,

				mt=β1mt-1 + (1 - β1) *[∂L/∂wt]
				vt=β2vt-1  +(1 - β2) *[∂L/∂wt]2 

The Metrics is used to monitor the training and testing steps. We decided to use accuracy and mean:

Accuracy = (number correct predictions)/(total number of predictions made) 
Mean = Sum/Total 

The key advantages of using this metric is its the easiest approach but not always the best and it works well if there are an equal number of samples belonging to each class. A key disadvantage of using the accuracy metric is it gives a false sense of achieving high accuracy when there is an unbalanced probability of each class occurring.

The Loss Function measures how accurate the model is during the training. Our Multiclass Cross Entropy Function:
-1/Ni=1Nyi log(pi),
where yi are the labels and,
 pi is the predicted probability of each label

Implementation

The code itself has comments on it explaining what we did. In addition, we wanted a brief text description of what was done. The first step was simply import our packages and get a path to the dataset we use from Kagglehub. After that we proceeded to take our images from the dataset, convert them to grayscale and resize them and split our images into a training array and a testing array as well as formatting our labels for our images into a training and testing array. We then created batches of our images to train. Finally, we make our model which is described in more detail in the discussion, and the code itself. And the training by using tensorflow GradientTape, Predictions which feeds the images to our model, and Loss which is in the discussion. The testing step is similar using predictions and loss but no GradientTape and setting training=Flase for predictions so our model is not in the training mode but just inference mode during the test step. Both the test and train functions are described in the code with comments.



Results

Our code ended up resulting in about a 7% accuracy for character which is better than the 2.8% accuracy of just randomly guessing. This means for every set of characters in a license plate it gets about 7% of the characters correct. However, this does not mean it gets 7% of the licenses correct since each license was 6 characters and furthermore needed to be in order. The code while working has some major issues. First was after numerous experimentation we we’re unable to get the accuracy per character we wanted and as a result we could not get an above .01% accuracy for the whole license plate. Due to this we get a model that has around a 7% accuracy of guessing a character out of A-Z 0-9 above the 2.8% of randomly guessing but not enough accuracy to get the character and place them in order for the license plate. Furthermore, as a result of low accuracy this we decided against attempting license plates with a state and trying to find the state the license is from.

Conclusion 

We have learned about the complexities of a neural network. The tensorflow library helped out a lot with the programming portion of the project, providing tools that helped us build and train the model.. The real challenge was figuring out how all the pieces fit together. Between the layers in the model build, the compiling of the model, and the training and testing, the data gets manipulated and rearranged in very complex ways. The data updates at every step so that it is in the optimal state to train the network to get the most desired results. 



References
baeldung. (2023, April 23). What is the purpose of a feature map in a convolutional neural network. Baeldung on Computer Science. https://www.baeldung.com/cs/cnn-feature-map 
Tariq, Farina. “Breaking Down the Mathematics Behind CNN Models: A Comprehensive Guide” Medium, 2 May 2023, medium.com/@beingfarina/breaking-down-the-mathematics-behind-cnn-models-a-comprehensive-guide-1853aa6b011e. Accessed 1 Dec. 2024.
“Basic classification: Classify images of clothing  |  TensorFlow Core” www.tensorflow.org/tutorials/keras/classification. Accessed 1 Dec. 2024.
Mishra, Aditya. “Metrics to Evaluate your Machine Learning Algorithm” Towards Data Science, 24 Feb. 2018, towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234. Accessed 1 Dec. 2024.
GeeksforGeeks, www.geeksforgeeks.org/. Accessed 1 Dec. 2024. 
“Tf.Keras.Losses.SparseCategoricalCrossentropy  :  Tensorflow V2.16.1.” TensorFlow, www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy. Accessed 1 Dec. 2024. 
Yazdani, Nick Navid. “License Plate Text Recognition Dataset.” Kaggle, 28 Dec. 2022, www.kaggle.com/datasets/nickyazdani/license-plate-text-recognition-dataset/data. 
Balachandran, Sandeep. “Machine Learning - Dense Layer” DEV Community, 19 Jan. 2020, dev.to/sandeepbalachandran/machine-learning-dense-layer-2m4n. Accessed 1 Dec. 2024.
Rastogi, Vaibhav. “Fully Connected Layer - Vaibhav Rastogi” Medium, 8 Sept. 2023, medium.com/@vaibhav1403/fully-connected-layer-f13275337c7c. Accessed 1 Dec. 2024.
“tf.keras.layers.Dropout  |  TensorFlow v2.16.1” www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout. Accessed 1 Dec. 2024.
Franco, Francesco. “The Softmax Activation Function with Keras” Medium, 13 Nov. 2024, ai.gopubby.com/the-softmax-activation-function-work-with-keras-8f674b4481a5. Accessed 1 Dec. 2024.
