import torch
import os
import cv2
import numpy as np
import onnxruntime as rt

class LandingDetector:

    def __init__(self):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'LandingDetector'
    def detect(self, image):
        # Load the YOLO model
        model_file = 'model/best.onnx'
        session = rt.InferenceSession(model_file)

        # Get the names of the model's input nodes
        input_names = [input.name for input in session.get_inputs()]

        # Preprocess the image
        input_size = (544, 544)
        resized_image = cv2.resize(image, input_size)
        input_data = np.transpose(resized_image, [2, 0, 1]) / 255.0
        input_data = np.expand_dims(input_data, axis=0).astype('float32')

        # Run inference
        output = session.run([], {input_names[0]: input_data})


        output_array = output[0]
        max_confidence_idx = np.argmax(output_array[:, :, 4])
        max_confidence_bbox = output_array[0, max_confidence_idx, :]
        print(max_confidence_bbox)
        x_min, y_min, x_max, y_max = max_confidence_bbox

        return 100, 100, 150, 150

        # Do something with the output
