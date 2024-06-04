from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class StrokeService:
    def checkStroke(self, data_dict):
        try:
            print("in stroke checking")
            # Get data
            gender = data_dict.get("gender")
            age = float(data_dict.get("age"))
            hypertension = float(data_dict.get("hypertension"))
            heart_disease = data_dict.get("heart_disease")
            ever_married = data_dict.get("ever_married")
            work_type = data_dict.get("work_type")
            residence_type = data_dict.get("residence_type")
            avg_glucose_level = float(data_dict.get("avg_glucose_level"))
            bmi = float(data_dict.get("bmi"))
            smoking_status = data_dict.get("smoking_status")

            # Preprocess data
            data = [[gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]]
            df = self.preprocessData(data)

            # load model from pickle file
            model_pkl_file = "./savedModels/stroke-rf.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
                # evaluate model
                print(df)
                y_predict = model.predict(df)
                print("y_predict: ", y_predict)
                probability = model.predict_proba(df)
                d = {
                    "stroke": int(y_predict[0]),
                    "probability": float(max(probability[0])*100)
                }
                
                return d
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        df = pd.DataFrame(data, columns=["gender","age","hypertension","heart_disease","ever_married",
                                         "work_type","residence_type","avg_glucose_level","bmi","smoking_status"])
        le = LabelEncoder()
        df['gender'] = le.fit_transform(df['gender'])
        df['smoking_status'] = le.fit_transform(df['smoking_status'])
        df['work_type'] = le.fit_transform(df['work_type'])
        df['ever_married'] = le.fit_transform(df['ever_married'])
        df['residence_type'] = le.fit_transform(df['residence_type'])

        df = df.drop(['gender','residence_type','work_type','smoking_status'], axis=1)

        return df


