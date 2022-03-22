class Synapse:
    weight = 0.0
    deltaWeight = 0.0
    # def __init__(self, w, dW):
    #     self.weight = w
    #     self.deltaWeight = dW

    def getWeight(self):
        return self.weight

    def getDeltaWeight(self):
        return self.deltaWeight

    def setDeltaWeight(self, __value):
        self.deltaWeight = __value

    def setWeight(self, __value):
        self.weight = __value

# syn = Synapse()
# syn.setWeight(80)
# print(syn.getWeight())
print(Synapse().weight)