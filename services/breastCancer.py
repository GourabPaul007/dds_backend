from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

class BreastCancerService:
    # def __init__(self):

    def checkBreastCancer(self, data_dict):
        try:
            print("in breast cancer checking")
            # Get data
            radius_mean = float(data_dict.get("radius_mean"))
            texture_mean = float(data_dict.get("texture_mean"))
            perimeter_mean = float(data_dict.get("perimeter_mean"))
            area_mean = data_dict.get("area_mean")
            smoothness_mean = float(data_dict.get("smoothness_mean"))
            compactness_mean = float(data_dict.get("compactness_mean"))
            concavity_mean = float(data_dict.get("concavity_mean"))
            concave_points_mean = float(data_dict.get("concave_points_mean"))
            symmetry_mean = float(data_dict.get("symmetry_mean"))
            fractal_dimension_mean = float(data_dict.get("fractal_dimension_mean"))
            radius_se = float(data_dict.get("radius_se"))
            texture_se = float(data_dict.get("texture_se"))
            perimeter_se = float(data_dict.get("perimeter_se"))
            area_se = float(data_dict.get("area_se"))
            smoothness_se = float(data_dict.get("smoothness_se"))
            compactness_se = float(data_dict.get("compactness_se"))
            concavity_se = float(data_dict.get("concavity_se"))
            concave_points_se = float(data_dict.get("concave_points_se"))
            symmetry_se = float(data_dict.get("symmetry_se"))
            fractal_dimension_se = float(data_dict.get("fractal_dimension_se"))
            radius_worst = float(data_dict.get("radius_worst"))
            texture_worst = float(data_dict.get("texture_worst"))
            perimeter_worst = float(data_dict.get("perimeter_worst"))
            area_worst = float(data_dict.get("area_worst"))
            smoothness_worst = float(data_dict.get("smoothness_worst"))
            compactness_worst = float(data_dict.get("compactness_worst"))
            concavity_worst = float(data_dict.get("concavity_worst"))
            concave_points_worst = float(data_dict.get("concave_points_worst"))
            symmetry_worst = float(data_dict.get("symmetry_worst"))
            fractal_dimension_worst = float(data_dict.get("fractal_dimension_worst"))


            # Preprocess data
            # data = [[6,148,72,35,0,33.6,0.627,50]]
            # data = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]
            data = [[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,
                     symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,
                     concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,
                     smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst,
                    ]]
            df = self.preprocessData(data)

            # load model from pickle file
            # model_pkl_file = "./savedModels/diabetes-rf.pkl"
            model_pkl_file = "./savedModels/bc-rf.pkl"
            with open(model_pkl_file, 'rb') as file:  
                model = pickle.load(file)
            
                # evaluate model
                # y_predict = model.predict([[1,89,66,23,94,28.1,0.167,21]])
                print(df)
                y_predict = model.predict(df)
                print("y_predict: ", y_predict)
                probability = model.predict_proba(df)
                d = {
                    "breast_cancer": int(y_predict[0]),
                    "probability": float(max(probability[0])*100)
                }
                
                return d
        
        except Exception as err:
            print(err)
    
    def preprocessData(self, data):
        df = pd.DataFrame(data, columns=["radius_mean",
                                         "texture_mean",
                                         "perimeter_mean",
                                         "area_mean",
                                         "smoothness_mean",
                                         "compactness_mean",
                                         "concavity_mean","concave_points_mean",
                                         "symmetry_mean","fractal_dimension_mean",
                                         "radius_se",
                                         "texture_se",
                                         "perimeter_se",
                                         "area_se",
                                         "smoothness_se",
                                         "compactness_se",
                                         "concavity_se",
                                         "concave_points_se",
                                         "symmetry_se",
                                         "fractal_dimension_se",
                                         "radius_worst",
                                         "texture_worst",
                                         "perimeter_worst",
                                         "area_worst",
                                         "smoothness_worst",
                                         "compactness_worst",
                                         "concavity_worst",
                                         "concave_points_worst",
                                         "symmetry_worst",
                                         "fractal_dimension_worst",
                                        ])
        return df


