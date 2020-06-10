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
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            # To load the model the tensorflow graph
            with tf.io.gfile.GFile(model_path, 'rb') as file:
                serialized_graph = file.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # INPUT : The image that gets fed to the graph
        self.input = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # OUTPUT : List of outputs we want to get from the API from each frame 
        self.output_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.output_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.output_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def predict(self,img):
        """
        Get the predicition results on 1 frame
        @ img : our img vector
        """


        # Pass the inputs and outputs to the session to get the results 
        # Change the dimension of the image to the correct one
        img_exp = np.expand_dims(img, axis=0)
        (boxes, scores, classes) = self.sess.run([self.output_boxes, self.output_scores, self.output_classes],feed_dict={self.input: img_exp})

        return (boxes, scores, classes)  
        

        # return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

        """
            def get_box(self,boxes,height,width):
                array_boxes = list()
                for i in range(boxes.shape[1]):
                    boxes_list.append((int(boxes[0,i,0] * height),int(boxes[0,i,1]*width),int(boxes[0,i,2] * height),int(boxes[0,i,3]*width)))
                return array_boxes
        """


    def close(self):
        self.sess.close()
        self.default_graph.close()
       

