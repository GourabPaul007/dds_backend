from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class ParkinsonsService:
    # def __init__(self):

    def checkParkinsons(self, data_dict):
        try:
            print("here")
            # Get data
            mdvp_fo_hz = float(data_dict.get("mdvp_fo_hz"))
            mdvp_fhi_hz = float(data_dict.get("mdvp_fhi_hz"))
            mdvp_flo_hz = float(data_dict.get("mdvp_flo_hz"))
            mdvp_jitter_in_percent = float(data_dict.get("mdvp_jitter_in_percent"))
            mdvp_jitter_abs_in_str = data_dict.get("mdvp_jitter_abs")
            if 'e' in mdvp_jitter_abs_in_str.lower():
                mdvp_jitter_abs = format(float(mdvp_jitter_abs_in_str),'f')
            else:
                mdvp_jitter_abs = float(mdvp_jitter_abs_in_str)
            mdvp_rap = float(data_dict.get("mdvp_rap"))
            mdvp_ppq = float(data_dict.get("mdvp_ppq"))
            jitter_ddp = float(data_dict.get("jitter_ddp"))
            mdvp_shimmer = float(data_dict.get("mdvp_shimmer"))
            mdvp_shimmer_db = float(data_dict.get("mdvp_shimmer_db"))
            shimmer_apq3 = float(data_dict.get("shimmer_apq3"))
            shimmer_apq5 = float(data_dict.get("shimmer_apq5"))
            mdvp_apq = float(data_dict.get("mdvp_apq"))
            shimmer_dda = float(data_dict.get("shimmer_dda"))
            nhr = float(data_dict.get("nhr"))
            hnr = float(data_dict.get("hnr"))
            rpde = float(data_dict.get("rpde"))
            dfa = float(data_dict.get("dfa"))
            spread1 = float(data_dict.get("spread1"))
            spread2 = float(data_dict.get("spread2"))
            d2 = float(data_dict.get("d2"))
            ppe = float(data_dict.get("ppe"))


            # Preprocess data
            # data = [[6,148,72,35,0,33.6,0.627,50]]
            # data = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]
            data = [[mdvp_fo_hz, mdvp_fhi_hz, mdvp_flo_hz, mdvp_jitter_in_percent,
                    mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer,
                    mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq,
                    shimmer_dda, nhr, hnr, rpde, dfa, spread1,
                    spread2, d2, ppe]]
            df = self.preprocessData(data)

            # load model from pickle file
            # model_pkl_file = "./savedModels/diabetes-rf.pkl"
            model_pkl_file = "./savedModels/parkinsons-rf.pkl"
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
        df = pd.DataFrame(data, columns=[
                                        'mdvp_fo_hz', 
                                        'mdvp_fhi_hz', 
                                        'mdvp_flo_hz', 
                                        'mdvp_jitter_in_percent',
                                        'mdvp_jitter_abs', 
                                        'mdvp_rap', 
                                        'mdvp_ppq', 
                                        'jitter_ddp', 
                                        'mdvp_shimmer',
                                        'mdvp_shimmer_db', 
                                        'shimmer_apq3', 
                                        'shimmer_apq5', 
                                        'mdvp_apq',
                                        'shimmer_dda', 
                                        'nhr', 
                                        'hnr', 
                                        'rpde', 
                                        'dfa', 
                                        'spread1',
                                        'spread2', 
                                        'd2', 
                                        'ppe']
                                        )
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


