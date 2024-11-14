#s5286336
import numpy as np
import matplotlib.pyplot as plt
import gzip
import sys
import time

#implementing the neural network class
class NeuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize


        self.weight1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.weight2 = np.random.randn(self.hiddenSize, self.outputSize)
        self.bias1 = np.random.randn(1, self.hiddenSize)
        self.bias2 = np.random.randn(1, self.outputSize)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoidPrime(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))
    
    def feedForward(self, X):
        self.Z1 = np.dot(X, self.weight1) + self.bias1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.weight2) + self.bias2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2
        
    def quadratic_cost_function(self, y, output):
        n = y.shape[0]
        return np.sum(0.5 * ((output - y) ** 2))/2

    def backPropagation(self, X, Y, output, learningRate):
        m = Y.shape[0]

        #output layer error
        d2 = (output - Y) * self.sigmoidPrime(self.Z2)
        dW2 = np.dot(self.A1.T, d2)
        dB2 = np.sum(d2, axis=0)

        #Hidden layer error
        d1 = np.dot(d2, self.weight2.T) * self.sigmoidPrime(self.Z1)
        dW1 = np.dot(X.T, d1)
        dB1 = np.sum(d1, axis=0)

        #update weights and biases
        self.weight1 -= learningRate * dW1 / m
        self.bias1 -= learningRate * dB1 / m
        self.weight2 -= learningRate * dW2 / m
        self.bias2 -= learningRate * dB2 / m

    def train(self, X_train, Y_train, X_test, Y_test, epochs, batch_size, learning_rate):
        history = []

        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                Y_batch = Y_train_shuffled[i:i + batch_size]

                epoch_cost = 0

                output = self.feedForward(X_batch)
                self.backPropagation(X_batch, Y_batch, output, learning_rate)

                batch_cost = self.quadratic_cost_function(Y_batch, output)
                epoch_cost += batch_cost
            epoch_cost /= (X_train.shape[0] / batch_size)

            accuracy = self.evaluate(X_test, Y_test)
            history.append(accuracy)
            print(f"Epoch {epoch + 1}/{epochs}, Cost: {epoch_cost:.4f}, Accuracy: {accuracy: .4f}")

        return history

    def evaluate(self, X, Y):
        output = self.feedForward(X)
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predictions == labels)

        return accuracy

#loading the dataset
def loadData(file):
    with gzip.open(file, 'rt') as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
    labels = data[:, 0].astype(int)
    input = data[:, 1:]/255.0

    return input, labels

def oneHot(labels, num_class=10):
    return np.eye(num_class)[labels]

#ploting the epoch vs accuracy graph
def plotAccuracy(history, title):
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True)
    plt.show()

#function to plot the samples from the dataset
def plotSamples(X_train):
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
    for i in range (5):
        ax[i].imshow(X_train[i].reshape(28, 28), cmap='gray')
        ax[i].axis('off')
    plt.show()

def main():
    if len(sys.argv) !=6:
        print("Usage: python nn.py NInput NHidden NOutput fashion-mnist_train.csv.gz fashion-mnist_test.csv.gz")
        return 

    NInput = int(sys.argv[1])
    NHidden = int(sys.argv[2])
    NOutput = int(sys.argv[3])
    trainFile = sys.argv[4]
    testFile = sys.argv[5]

    X_train, Y_train = loadData(trainFile)
    X_test, Y_test = loadData(testFile)

    Y_train = oneHot(Y_train, NOutput)
    Y_test = oneHot(Y_test, NOutput)

    plotSamples(X_train)

    nn = NeuralNetwork(NInput, NHidden, NOutput)

    #task 1 - training dataset with the following settings: epoch = 30, mini-batch size = 20, 𝜂 = 3.
    epochs = 30
    batch_size = 20
    learningRate = 3

    history = nn.train(X_train, Y_train, X_test, Y_test, epochs, batch_size, learningRate)
    testAccuracy = nn.evaluate(X_test, Y_test)
    print(f"Maximum test accuracy: {testAccuracy:.4f}")
    print()

    plotAccuracy(history, "Test Accuracy Vs Epoch")
    
    #task 2 - different learning rates 𝜂 = 0.001, 0.01, 1.0, 10, 100
    learningRates = [0.001, 0.01, 1.0, 10, 100]
    maxAcc = []

    plt.figure(figsize=(10, 6))
    for learningRate in learningRates:
        nn = NeuralNetwork(NInput, NHidden, NOutput)
        epochs = 30
        batch_size = 20
        history = nn.train(X_train, Y_train, X_test, Y_test, epochs, batch_size, learningRate)
        print()
        maxAccuracy = max(history)
        maxAcc.append(maxAccuracy)

        plt.plot(history, label=f'Learning Rate: {learningRate}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Epoch for different learning rates')
    plt.legend()
    plt.grid(True)
    plt.show()

    print()
    for i, learningRate in enumerate(learningRates):
        print(f"Maximum accuracy for learning rates {learningRate}: {maxAcc[i]:.4f}")


#task 3 different mini-batch sizes = 1, 5, 20, 100, 300
    print()
    batchSizes = [1, 5, 20, 100, 300]
    maxAcc = []
    times = []

    for batchSize in batchSizes:
        nn = NeuralNetwork(NInput, NHidden, NOutput)
        epochs = 30
        learningRate = 3

        start_time = time.time()
        history = nn.train(X_train, Y_train, X_test, Y_test, epochs, batch_size, learningRate)
        end_time = time.time()
        print()
        maxAccuracy = max(history)
        maxAcc.append(maxAccuracy)
        times.append(end_time-start_time)

    plt.figure(figsize=(10, 6))
    plt.plot(batchSizes, maxAcc, marker='o')    
    plt.xlabel('Mini-batch size')
    plt.ylabel('Maximum test Accuracy')
    plt.title('Maximum test Accuracy vs Mini-batch')
    plt.grid(True)
    plt.show()

    print()
    for i, batchSize in enumerate(batchSizes):
        print(f"Maximum accuracy for mini batche size {batchSize}: {maxAcc[i]:.4f}, Traning time: {times[i]:.2f} seconds")

    print()
    print(f"The batch size achieves the highest accuracy: {batchSizes[np.argmax(maxAcc)]}")
    print(f"The slowest batch size: {batchSizes[np.argmax(times)]}")

if __name__ == "__main__":
    main()









