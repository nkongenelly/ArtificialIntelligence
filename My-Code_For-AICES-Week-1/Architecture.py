class Architecture():
    # features
    description = ''

    def __init__(self, description):
        self.descriptionList = [description.split(",")]

        # for dLI in range(0, len(self.descriptionList) - 1, 1):
        #    self.add(int(self.descriptionList[dLI]))

    def __len__(self):
        return len(self.descriptionList)