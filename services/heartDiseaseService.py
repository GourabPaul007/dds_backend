from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class HeartDiseaseService:
    # def __init__(self):

    def checkHeartDisease(self, data_dict):
        try:
            print("here")
            # Get data
            age = float(data_dict.get("age"))
            sex = float(data_dict.get("sex"))
            chest_pain_type = float(data_dict.get("chest_pain_type"))
            resting_bp = float(data_dict.get("resting_bp"))
            cholesterol = float(data_dict.get("cholesterol"))
            fasting_bs = float(data_dict.get("fasting_bs"))
            resting_ecg = float(data_dict.get("resting_ecg"))
            max_hr = float(data_dict.get("max_hr"))
            exercise_angina = float(data_dict.get("exercise_angina"))
            oldpeak = float(data_dict.get("oldpeak"))
            st_slope = float(data_dict.get("st_slope"))

            # Preprocess data
            data = [[age,sex,chest_pain_type,resting_bp,cholesterol,fasting_bs,resting_ecg,max_hr,exercise_angina,oldpeak,st_slope]]
            df = self.preprocessData(data)

            # load model from pickle file
            model_pkl_file = "./savedModels/hd-gbc.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
                # evaluate model
                # y_predict = model.predict([[1,89,66,23,94,28.1,0.167,21]])
                print(df)
                y_predict = model.predict(df)
                print("y_predict: ", y_predict)
                probability = model.predict_proba(df)
                d = {
                    "parkinsons": int(y_predict[0]),
                    "probability": float(max(probability[0])*100)
                }
                
                return d
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        df = pd.DataFrame(data, columns=["age","sex","chest_pain_type","resting_bp","cholesterol","fasting_bs",
                                         "resting_ecg","max_hr","exercise_angina","oldpeak","st_slope",])
            # 119.992,
            # 157.302,
            # 74.997,
            # 0.00784,
            # 7e-05,
            # 0.0037,
            # 0.00554,
            # 0.01109,
            # 0.04374,
            # 0.426,
            # 0.02182,
            # 0.0313,
            # 0.02971,
            # 0.06545,
            # 0.02211,
            # 21.033,
            # 0.414783,
            # 0.815285,
            # -4.813031,
            # 0.266482,
            # 2.301442,
            # 0.284654
        return df

