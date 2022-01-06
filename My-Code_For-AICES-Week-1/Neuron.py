import Synapse
import random
import math

# print(Synapse.Synapse().weight)
# print(random.uniform(0.0, 1.0))
class Neuron:
    # eta = 0.0
    # alpha = 0.0
    gradient = 0.0
    outcome = 0.0
    # neuronId = 0
    # numberOfWeightsFromNextNeuron = 0
    weights = []

    def __init__(self, eta=0.0, alpha=0.0, numberOfWeightsFromNextNeuron=0, neuronId=0,):
        self.eta = eta
        self.alpha = alpha
        self.numberOfWeightsFromNextNeuron = numberOfWeightsFromNextNeuron
        self.neuronId = neuronId
        self.gradient = 0.0

        for wI in range(0, numberOfWeightsFromNextNeuron - 1, 1):
            self.weights.append(Synapse.Synapse())
            self.weights[wI] = random.uniform(0.0, 1.0)

    # methods
    # Observe stuff about neuronn
    def getOutcome(self):
        return self.outcome

    def getGradient(self):
        return self.gradient

    def getWeights(self):
        return self.weights

    def getActivation(self, value):
        return math.tanh(value)

    def getPrimeActivation(self, value):
        return 1 -  math ** (math.tanh(value), 2)

    def getDistributedWeight(self, nextLayer):
        sigma = 0.0

        # this neuron's weights * the gradient of other neurons
        for nLI in range(0, len(nextLayer) - 2, 1):
            sigma += self.getWeights()[nLI].Synapse.Synapse().getWeight() * nextLayer[nLI].getGradient()

        return sigma

    # modify stuff about neuron
    def setOutcome(self, value):
        self.outcome = value

    def setGradient(self, value):
        self.gradient = value

    def setHiddenGradient(self, nextLayer):
        delta = self.getDistributedWeight(nextLayer)
        self.setGradient(self.getPrimeActivation(self.outcome) * delta)

    def setOutcomeGradient(self, target):
        delta = target - self.outcome
        self.setGradient(self.getPrimeActivation(self.outcome) * delta)

    def doForwardPropagation(self, priorLayer):
        sigma = 0.0

        # other layer weights * other layer outcome
        for pLI in range(0, len(priorLayer) - 1, 1):
            sigma += priorLayer[pLI](self.getWeights())[self.neuronId] * Synapse.Synapse().getWeight() * priorLayer[pLI](self.getOutcome())

        self.setOutcome(self.getActivation(sigma))

    def updateWeight(self, priorLayer):
        for pLI in range(0, len(priorLayer), 1):
            priorDeltaWeight = priorLayer[pLI](self.getWeights())[self.neuronId](Synapse.Synapse().getDeltaWeight())

            # formula: (eta * this gradient * outcome) + (alpha * priorDeltaWeight)
            newDeltaWeight = (self.eta * self.getGradient() * priorLayer[pLI](self.getOutcome())) + (self.alpha * priorDeltaWeight)

            # set our delta weight
            priorLayer.get(pLI)(self.getWeights())[self.neuronId](Synapse.Synapse().setDeltaWeight(newDeltaWeight))

            # set our weights
            priorLayer[pLI](self.getWeights())[self.neuronId](Synapse.Synapse().setWeight(priorLayer[pLI])(self.getWeights())[self.neuronId](Synapse.Synapse().getWeight()) + newDeltaWeight)