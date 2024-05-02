import base64
import numpy as np
import tensorflow as tf
import os
import cv2

class TuberculosisService:
    # def __init__(self):

    def checkTuberculosis(self, data):
        try:
            # Preprocess data
            imgArr = self.preprocessData(data) # returns ndarry from image

            # load model from h5 file
            model_file_path = 'savedModels/tb-cnn.h5'
            model = tf.keras.models.load_model(model_file_path)

            # evaluate model
            y_predict = model.predict(imgArr[None,:,:])
            pred = y_predict[0]
            print("y_predict: ", pred)
            btDict = {
                0: 'no_tuberculosis',
                1: 'yes_tuberculosis',
            }
            res = 1 if pred>0.5 else 0
                
            return int(res)
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        imgArr = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        return imgArr

