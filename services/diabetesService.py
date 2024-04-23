from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class DiabetesService:
    # def __init__(self):

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
            # data = [[6,148,72,35,0,33.6,0.627,50]]
            # data = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]
            data = [[pregnancies, glucose, bloodPressure, insulin, bmi, diabetesPedigreeFunction, age]]
            df = self.preprocessData(data)

            # load model from pickle file
            # model_pkl_file = "./savedModels/diabetes-rf.pkl"
            model_pkl_file = "./savedModels/DM_RandomF.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
                # evaluate model
                # y_predict = model.predict([[1,89,66,23,94,28.1,0.167,21]])
                print(df)
                y_predict = model.predict(df)
                print("y_predict: ", y_predict)
                
                return int(y_predict[0])
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        scaler = StandardScaler()
        scaler.fit_transform(data)
        # df = pd.DataFrame(data, columns=['pregnancies','glucose','blood_pressure','skin_thickness','insulin','bmi','diabetes_pedigree_function','age'])
        df = pd.DataFrame(data, columns=["pregnancies","glucose","blood_pressure","insulin","bmi","diabetes_pedigree_function","age"])
        return df
