import numpy as np
import tensorflow as tf
import cv2

class CovidService:
    # def __init__(self):

    def checkCovid(self, data):
        try:
            # Preprocess data
            imgArr = self.preprocessData(data) # returns ndarry from image

            # load model from h5 file
            model_file_path = 'savedModels/covid-cnn.h5'
            model = tf.keras.models.load_model(model_file_path)

            # evaluate model
            y_predict = model.predict(imgArr[None,:,:])
            res = y_predict[0]
            print("y_predict: ", res)
            covidDict = {
                0: 'covid',
                1: 'lung_opacity',
                2: 'normal',
                3: 'viral_pneumonia',
            }
            covidDictPercentage = {
                # 'c0_glioma':1.0,
                # 'c1_meningioma':0.0,
                # 'c2_notumor':0.0,
                # 'c3_pituitary':0.0,
            }
            for i in range(len(res)):
                covidDictPercentage[covidDict[i]] = str(res[i])
            result = {"output": covidDict[np.argmax(res)], "probabilities": covidDictPercentage}
            print(result)

            return result
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        imgArr = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
        resized = cv2.resize(imgArr, (100,100), interpolation = cv2.INTER_LINEAR)
        return resized

