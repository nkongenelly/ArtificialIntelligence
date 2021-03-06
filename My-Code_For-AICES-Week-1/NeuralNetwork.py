from Layers import Layers
from Layer import Layer
from Architecture import Architecture
from Neuron import Neuron
from pprint import pprint

class NeuralNetwork:
    # features
    eta = 0.2
    alpha = 0.5

    layers = Layers()
    architecture = Architecture("2,2,1")

    def __init__(self):
        for lSI in range(0, len(self.architecture) -1, 1):
            self.layers.append(Layer())

            for lI in range(0, self.architecture[lSI], 1):
                # Neuron(double eta, double alpha, int numberOfWeightsFromNextNeuron, int neuronId )
                # FORMULA: IF lSI + 1 < architecture.size() arch(lSI + 1) else 0 ....a if condition else b
                numberOfWeightsFromNextNeuron = self.architecture[lSI + 1] if lSI + 1 < len(self.architecture) else 0

                newNeuron = Neuron(self.eta, self.alpha, numberOfWeightsFromNextNeuron, lI)

                self.layers[lSI].append(newNeuron)

                self.layers[lSI][len(self.layers[lSI]) - 1] * (Neuron().setOutcome(1.0))

    # do forward propagation
    def doForwardPropagation(self, inputs):
        # pass inputs to 1st layer of neural network
        print("doForwardPropagation inputs = ", len(inputs))
        print("doForwardPropagation self.layers before = ", self.layers)
        iI = 0
        # print("index inputs ? = ", self.layers[0])
        while (iI < len(inputs)):
            # print("index iI ? = ", iI)
            self.layers[0][iI].setOutcome(inputs[iI])
            iI += 1
        #
        lSI = 1
        while (lSI < len(self.architecture)):
            priorLayer = self.layers[lSI - 1]
            lI = 0
            while (lI < self.architecture[lSI]):
                self.layers[lSI][lI].doForwardPropagation(priorLayer)
                lI += 1
            lSI += 1
        # for iI in range(0, len(inputs) - 1, 1):
        #     self.layers[0][iI] * (Neuron().setOutcome(inputs[iI]))
        #     print("doForwardPropagation layers = ")
        #     print(self.layers)
        #
        # for lSI in range(1,len(self.architecture) - 1, 1):
        #     priorLayer = self.layers[lSI - 1];
        #     for lI in range(0, self.architecture[lSI] - 1, 1):
        #         self.layers[lSI][lI] * (Neuron().doForwardPropagation(priorLayer))

    # backward propagation
    def doBackwardPropagation(self, target):
        # set outcome gradient
        outcomeNeuron = self.layers[len(self.layers) - 1][0]
        outcomeNeuron(Neuron().setOutcomeGradient(target))

        # set hidden gradient
        for lSI in range(len(self.layers) - 2, 1, -1):
            currentLayer = self.layers(lSI)
            nextLayer = self.layers[lSI + 1]

            for lI in range(0, len(currentLayer) - 1, 1):
                currentLayer[lI] * (Neuron().setHiddenGradient(nextLayer))

        # update weights
        for lSI in range(len(self.layers) -1, 1, -1):
            currentLayer = self.layers[lSI]
            priorLayer = self.layers[lSI - 1]
            for lI in range(0, len(currentLayer) - 2, 1):
                currentLayer[lI] * (Neuron().updateWeight(priorLayer))

    # get outcome of neural network
    def getOutcome(self):
        # print("layers = ", self.layers[0][0])
        return self.layers[len(self.layers) - 1][0] * (Neuron().getOutcome())