import numpy as np
import tensorflow as tf
import cv2
import time


class Model:
    """
    Class that contains the model and all its functions
    """
    def __init__(self, model_path):
        """
        Initialization function
        @ model_path : path to the model 
        """

        # Declare detection graph
        self.detection_graph = tf.Graph()
        # Load the model into the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as file:
                serialized_graph = file.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Create a session from the detection graph
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def predict(self,img):
        """
        Get the predicition results on 1 frame
        @ img : our img vector
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        img_exp = np.expand_dims(img, axis=0)
        # Pass the inputs and outputs to the session to get the results 
        (boxes, scores, classes) = self.sess.run([self.detection_graph.get_tensor_by_name('detection_boxes:0'), self.detection_graph.get_tensor_by_name('detection_scores:0'), self.detection_graph.get_tensor_by_name('detection_classes:0')],feed_dict={self.detection_graph.get_tensor_by_name('image_tensor:0'): img_exp})
        return (boxes, scores, classes)  

