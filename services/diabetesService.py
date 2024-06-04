from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class DiabetesService:
    def checkDiabetes(self, data_dict):
        try:
            # Get data
            age = float(data_dict.get("age"))
            pregnancies = float(data_dict.get("pregnancies"))
            glucose = float(data_dict.get("glucose"))
            bloodPressure = float(data_dict.get("blood_pressure"))
            skinThickness = float(data_dict.get("skin_thickness"))
            insulin = float(data_dict.get("insulin"))
            bmi = float(data_dict.get("bmi"))
            diabetesPedigreeFunction = float(data_dict.get("diabetes_pedigree_function"))

            # Validate data
            if(pregnancies<0 and glucose<0 and bloodPressure<0 and skinThickness<0 and insulin<0 and bmi<0 and diabetesPedigreeFunction<0 and age<0):
                return -1

            # Preprocess data
            data = [[pregnancies, glucose, bloodPressure, insulin, bmi, diabetesPedigreeFunction, age]]
            df = self.preprocessData(data)

            # load model from pickle file
            model_pkl_file = "./savedModels/DM_RandomF.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
                # evaluate model
                print(df)
                y_predict = model.predict(df)
                print("y_predict: ", y_predict)
                probability = model.predict_proba(df)
                d = {
                    "diabetes": int(y_predict[0]),
                    "probability": float(max(probability[0])*100)
                }
                
                return d
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        # sc = StandardScaler()
        # data = scaler.fit_transform(data)
        print(data)
        with open('./scalers/scaler_dm.pkl','rb') as f:
            sc = pickle.load(f)
            data = sc.transform(data)
            df = pd.DataFrame(data, columns=["pregnancies","glucose","blood_pressure","insulin","bmi","diabetes_pedigree_function","age"])
            return df
