import numpy as np
import tensorflow as tf
import cv2

class BrainTumorService:
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
                0: 'c0_glioma',
                1: 'c1_meningioma',
                2: 'c2_notumor',
                3: 'c3_pituitary'
            }
            btDictPercentage = {
                # 'c0_glioma':1.0,
                # 'c1_meningioma':0.0,
                # 'c2_notumor':0.0,
                # 'c3_pituitary':0.0,
            }
            for i in range(len(res)):
                btDictPercentage[btDict[i]] = str(res[i])
            result = {"output": btDict[np.argmax(res)], "probabilities": btDictPercentage}
            print(result)
                
            return result
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        imgArr = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
        return imgArr

