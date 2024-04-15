import base64
import numpy as np
import tensorflow as tf
import os
import cv2

class BrainTumorService:
    # def __init__(self):

    def checkBrainTumor(self, data):
        try:
            # Preprocess data
            imgArr = self.preprocessData(data) # returns ndarry from image

            # load model from h5 file
            model_file_path = 'savedModels/bt-cnn.h5'
            model = tf.keras.models.load_model(model_file_path)

            # evaluate model
            y_predict = model.predict(imgArr[None,:,:])
            res = y_predict[0]
            print("y_predict: ", res)
            btDict = {
                0: 'glioma',
                1: 'meningioma',
                2: 'notumor',
                3: 'pituitary'
            }
            argmax = np.argmax(res)
            print(argmax, btDict[argmax])
                
            return int(argmax)
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        imgArr = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
        return imgArr

