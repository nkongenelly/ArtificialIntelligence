class Architecture():
    # features
    description = ''

    def __init__(self, description):
        descriptionList = description.split(",")
        self.descriptionList = descriptionList

        for dLI in range(0, len(description) - 1, 1):
           descriptionList += descriptionList[dLI]

    def __len__(self):
        return len(self.descriptionList)