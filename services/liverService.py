from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class LiverService:
    # def __init__(self):

    def checkLiver(self, data_dict):
        try:
            # Get data
            age = float(data_dict.get("age"))
            gender = data_dict.get("gender")
            total_Bilirubin = float(data_dict.get("total_bilirubin"))
            direct_Bilirubin = float(data_dict.get("direct_bilirubin"))
            alkaline_Phosphotase = float(data_dict.get("alkaline_phosphotase"))
            alamine_Aminotransferase = float(data_dict.get("alamine_aminotransferase"))
            aspartate_Aminotransferase = float(data_dict.get("aspartate_aminotransferase"))
            total_Protiens = float(data_dict.get("total_protiens"))
            albumin = float(data_dict.get("albumin"))
            albumin_and_Globulin_Ratio = float(data_dict.get("albumin_and_globulin_ratio"))
            print("age",age)

            # Validate data
            if(age<0 and gender<0 and total_Bilirubin<0 and direct_Bilirubin<0 and alkaline_Phosphotase<0 and alamine_Aminotransferase<0 and aspartate_Aminotransferase<0 and total_Protiens<0 and albumin<0 and albumin_and_Globulin_Ratio<0):
                return -1

            # Preprocess data
            if(gender=="male"):
                gender = 1
            else:
                gender = 0
            data = [[age, gender, total_Bilirubin, direct_Bilirubin, alkaline_Phosphotase, alamine_Aminotransferase, aspartate_Aminotransferase, total_Protiens, albumin, albumin_and_Globulin_Ratio]]
            df = self.preprocessData(data)

            # load model from pickle file
            model_pkl_file = "./savedModels/liver-rf.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
            # evaluate model
            # y_predict = model.predict([[1,89,66,23,94,28.1,0.167,21]])
            y_predict = model.predict(df)
            print("y_predict: ", y_predict)
            
            return int(y_predict[0])
        
        except Exception as err:
            print("err",err)
    
    def preprocessData(self, data):
        scaler = StandardScaler()
        scaler.fit_transform(data)
        df = pd.DataFrame(data, columns=["age", "gender", "total_bilirubin", "direct_bilirubin", "alkaline_phosphotase", "alamine_aminotransferase", "aspartate_aminotransferase", "total_protiens", "albumin", "albumin_and_globulin_ratio"])
        return df
