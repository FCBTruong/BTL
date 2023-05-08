import cv2
import numpy as np

class LandingDetector:

    def __init__(self):
        """
        You should hard fix all the requirement parameters
        """
        self.name = 'LandingDetector'

        # Load the YOLO model
        model_file = 'model/best.onnx'
        self.yolo_net = cv2.dnn.readNetFromONNX(model_file)

    def detect(self, image):
        # Preprocess the image
        input_size = (544, 544)
        resized_image = cv2.resize(image, input_size)
        input_data = np.transpose(resized_image, [2, 0, 1]) / 255.0
        input_data = np.expand_dims(input_data, axis=0).astype('float32')

        # Set the input and output nodes for the YOLO network
        self.yolo_net.setInput(input_data)
        output_names = self.yolo_net.getUnconnectedOutLayersNames()

        # Run inference
        output = self.yolo_net.forward(output_names)

        output_array = output[0]
        max_confidence_idx = np.argmax(output_array[0, :, 4])
        max_confidence_bbox = output_array[0, max_confidence_idx, :]
        print(max_confidence_bbox)
        x_center, y_center, width, height, _, _ = max_confidence_bbox

        print("result ", x_center / 540, y_center / 540, width / 540, height / 540)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        return x_min, y_min, x_max, y_max

        # Do something with the output
