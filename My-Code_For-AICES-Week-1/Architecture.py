class Architecture:
    # features
    description = ''

    def __init__(self, description):
        descriptionList = description.split(",")

        for dLI in range(0, len(description) - 1, 1):
            sum(int(descriptionList[dLI]))
