import torch

class LandingDetector:

    def __init__(self):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'LandingDetector'
        self.model = torch.load('model/best.pt')

    def detect(self, img):
        output = self.model(input)
        return output
        #return 100, 100, 150, 150


