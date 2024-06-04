from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class LungCancerService:
    # def __init__(self):

    def checkLungCancer(self, data_dict):
        try:
            print("here")
            # Get data
            age = float(data_dict.get("age"))
            gender = float(data_dict.get("gender"))
            air_pollution = float(data_dict.get("air_pollution"))
            alcohol_use = float(data_dict.get("alcohol_use"))
            dust_allergy = float(data_dict.get("dust_allergy"))
            occupational_hazards = float(data_dict.get("occupational_hazards"))
            genetic_risk = float(data_dict.get("genetic_risk"))
            chronic_lung_disease = float(data_dict.get("chronic_lung_disease"))
            balanced_diet = float(data_dict.get("balanced_diet"))
            obesity = float(data_dict.get("obesity"))
            smoking = float(data_dict.get("smoking"))
            passive_smoker = float(data_dict.get("passive_smoker"))
            chest_pain = float(data_dict.get("chest_pain"))
            coughing_of_blood = float(data_dict.get("coughing_of_blood"))
            fatigue = float(data_dict.get("fatigue"))
            weight_loss = float(data_dict.get("weight_loss"))
            shortness_of_breath = float(data_dict.get("shortness_of_breath"))
            wheezing = float(data_dict.get("wheezing"))
            swallowing_difficulty = float(data_dict.get("swallowing_difficulty"))
            clubbing_of_finger_nails = float(data_dict.get("clubbing_of_finger_nails"))
            frequent_cold = float(data_dict.get("frequent_cold"))
            dry_cough = float(data_dict.get("dry_cough"))
            snoring = float(data_dict.get("snoring"))

            # Preprocess data
            data = [[age,gender,air_pollution,alcohol_use,dust_allergy,occupational_hazards,genetic_risk,chronic_lung_disease,
                     balanced_diet,obesity,smoking,passive_smoker,chest_pain,coughing_of_blood,fatigue,weight_loss,shortness_of_breath,
                     wheezing,swallowing_difficulty,clubbing_of_finger_nails,frequent_cold,dry_cough,snoring]]
            df = self.preprocessData(data)

            # load model from pickle file
            model_pkl_file = "./savedModels/lc-lr.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
                # evaluate model
                # y_predict = model.predict([[1,89,66,23,94,28.1,0.167,21]])
                print(df)
                y_predict = model.predict(df)
                print("y_predict: ", y_predict)
                probability = model.predict_proba(df)
                d = {
                    "lung_cancer": int(y_predict[0]),
                    "probability": float(max(probability[0])*100)
                }
                
                return d
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        df = pd.DataFrame(data, columns=["age","gender","air_pollution","alcohol_use","dust_allergy","occupational_hazards",
                                         "genetic_risk","chronic_lung_disease","balanced_diet","obesity","smoking","passive_smoker",
                                         "chest_pain","coughing_of_blood","fatigue","weight_loss","shortness_of_breath","wheezing",
                                         "swallowing_difficulty","clubbing_of_finger_nails","frequent_cold","dry_cough","snoring"])
        return df


