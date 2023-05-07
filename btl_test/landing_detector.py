import torch
import os
import cv2
import numpy as np

class LandingDetector:

    def __init__(self):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'LandingDetector'
        print('load weight')
        # Add the directory containing the models module to the PYTHONPATH

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 'model/best.pt')

        print('done load')

    def detect(self, img):
        # Resize image to match input dimensions of YOLOv5 model
        image = cv2.resize(img, (544, 544))

        # Convert image to NumPy array
        image_array = np.array(image).transpose(2, 0, 1)

        # Convert NumPy array to PyTorch tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()

        output = self.model(image_tensor)
        # Extract object positions (X, Y), width, and height for the object with highest confidence score

        # Get the top object detection result with the highest confidence score
        top_result = max(output[0], key=lambda x: x[4])

        # Extract object positions (X, Y), width, and height for the top object
        object_class = top_result[8]
        x, y, w, h = top_result[:4]
        print(x, y, w, h)
        print('finish detect')
        return output
        #return 100, 100, 150, 150


