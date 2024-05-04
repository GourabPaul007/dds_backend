import base64
import numpy as np
import tensorflow as tf
import os
import cv2

class AlzheimersService:
    # def __init__(self):

    def checkAlzheimers(self, data):
        try:
            # Preprocess data
            imgArr = self.preprocessData(data) # returns ndarry from image

            # load model from h5 file
            model_file_path = 'savedModels/alzheimers-imb-cnn.h5'
            model = tf.keras.models.load_model(model_file_path)

            # evaluate model
            y_predict = model.predict(imgArr[None,:,:])
            res = y_predict[0]
            print("y_predict: ", res)
            alDict = {
                0: 'c0_mild_demented',
                1: 'c1_moderate_demented',
                2: 'c2_non_demented',
                3: 'c3_very_mild_demented'
            }
            alDictPercentage = {
                # 'c0_mild_demented':1.0,
                # 'c1_moderate_demented':0.0,
                # 'c2_non_demented':0.0,
                # 'c3_very_mild_demented':0.0,
            }
            for i in range(len(res)):
                alDictPercentage[alDict[i]] = str(res[i])
            result = {"output": alDict[np.argmax(res)], "probabilities": alDictPercentage}
            print(result)
                
            return result
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        imgArr = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
        return imgArr

