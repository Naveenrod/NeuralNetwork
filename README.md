# NeuralNetwork
Software design
The software implemented a simple neural network for image classification using Fashion-MNIST dataset. The neural network consists of an input layer, a hidden layer, and an output layer. The code includes functions for loading and preprocessing the data  defining the neural network architecture, training the model using backpropagation and evaluating its performance. Moreover, there are utility functions for visualising training progress ad making predictions on real test images.

NeuralNetwork class
Attributes
inputSize: Number of input neurons.
hiddenSize: Number of neurons in the hidden layer.
outputSize: Number of neurons in the output layer.
weight1: weight matrix connecting input layer to hidden layer.
weight2: weight matrix connecting hidden layer to output layer.
bias1: bias vector for the hidden layer.
Bias2: bias vector for the output layer.

Methods
__init__ function:
	Initialises the neural network with input size, hidden layer size and output size. Randomly initialises weight and biases.

Sigmoid function:
	Computes the sigmoid activation for the given input.

sigmoidPrime function:
	computes the derivative of the sigmoid activation function.

Quadratic_cost_function function:
	Computes the quadratic cost function between the predicted output and the actual output.

feedforward function:
	Performs the forward pass of the neural network.

backpropagation function:
	Performs backpropagation to update weights and biases.

Train function:
	Trains the neural network using gradient descent and evaluate its performance over multiple epochs.

Evaluate function:
	Evaluates the accuracy of the neural network on a given dataset.

Utility Functions
Attributes
X_train: Input training data (features).
Y_train: Output training data(labels).
X_test: Input testing data.
Y_test: Output testing data.

Methods 
loadData function: 
Loads data from gzipped csv file, extracts labels and normalises input data.
oneHot function:
	Converts integer labels into on-hot encoded vectors.
plotAccuracy function:
	Plot the accuracy history over epochs.
Plotsamples function:
	Plots a sample of input images form the training dataset.
Main function:
	Entry point of the program, read command-line arguments, loads data, initialises the neural network, trains it and evaluate its performance. 

Data Structures
Numpy Array:
	Used extensively to store and handle data. History – list to store accuracy history during training. 

Test Results and Explanations 
The implemented neural network was trained using the fashion-mnist dataset. The neutral network was tested in using various parameters. The neural network had 784 neurons in the input layer, 30 neurons in the hidden layer and 10 neurons in the output layer. The dataset was loaded successfully below is an image consisting of some data from that dataset.

Test accuracy over epochs:
Initially the accuracy starts low and gradually increasing with each epoch. This indicates the model is learning from the training data. The neural network was trained over 30 epochs, achieving a maximum test accuracy of 86.35%. the model demonstrates effective learning with accurate steadily improving throughout the training process.

Learning rates:
Lower learning (0.001) rates lead to slower convergence and lower final accuracy. This is evident from the maximum accuracy achieved in these experiments, which is the lowest among all learning rates tested. Moderate learning (0.01) rates seem to strike a balance between convergence speed and final accuracy. This is reflected in the relatively higher maximum accuracy compared to lower learning rates. Higher learning rates (1, 10) result in faster convergence but may fluctuate around the optimal solutions, leading to reduced final accuracy. However, in the experiment a learning rate of 1.0 resulted in highest accuracy, suggesting that a higher learning rate might be beneficial. Extremely high learning rate (100) lead to divergence where the model fails to converge to a solution as evidenced by consistently low accuracy across epochs.

Mini batches:
Based on the experiments it appeared that the highest accuracy was achieved with a mini-batch size of 1, reaching an accuracy of 0.8694. However, this came at the cost of increased training time taking 29.06 seconds. On the other hand, the slowest batch size was also 1. So, while a smaller batch size can lead to better accuracy, it often comes with increased training time. 
In this implementation the performance with various mini-batches sizes, smaller batches generally lead to higher accuracy but longer training times. Conversely, larger batches resulted in slightly lower accuracy but faster training. The optimal choice depends on the specific requirement, balancing between accuracy and efficiency.

Different Hyper-parameter settings:
To explore different hyperparameter settings, I varied the number of epochs and mini-batch sizes. Below are some test accuracies achieved along with the corresponding hyperparameters.
Test accuracy: 87.07% (Number of epochs – 100, Mini-Batch size – 20)
Test accuracy: 86.54% (Number of epochs – 50, Mini-Batch size – 20)
Test accuracy: 58.5% (Number of epochs – 100, Mini-Batch size – 1)
Test accuracy: 87.5% (Number of epochs – 100, Mini-Batch size – 10)

