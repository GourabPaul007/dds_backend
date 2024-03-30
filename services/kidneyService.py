from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd

class KidneyService:
    # def __init__(self):

    def checkKidney(self, data_dict):
        try:
            # Get data
            age = float(data_dict.get("age"))
            blood_pressure = data_dict.get("blood_pressure")
            specific_gravity = float(data_dict.get("specific_gravity"))
            albumin = float(data_dict.get("albumin"))
            sugar = float(data_dict.get("sugar"))
            red_blood_cells = data_dict.get("red_blood_cells")
            pus_cell = data_dict.get("pus_cell")
            pus_cell_clumps = data_dict.get("pus_cell_clumps")
            bacteria = data_dict.get("bacteria")
            blood_glucose_random = float(data_dict.get("blood_glucose_random"))
            blood_urea = float(data_dict.get("blood_urea"))
            serum_creatinine = float(data_dict.get("serum_creatinine"))
            sodium = float(data_dict.get("sodium"))
            potassium = float(data_dict.get("potassium"))
            haemoglobin = float(data_dict.get("haemoglobin"))
            packed_cell_volume = float(data_dict.get("packed_cell_volume"))
            white_blood_cell_count = float(data_dict.get("white_blood_cell_count"))
            red_blood_cell_count = float(data_dict.get("red_blood_cell_count"))
            hypertension = data_dict.get("hypertension")
            diabetes_mellitus = data_dict.get("diabetes_mellitus")
            coronary_artery_disease = data_dict.get("coronary_artery_disease")
            appetite = data_dict.get("appetite")
            peda_edema = data_dict.get("peda_edema")
            aanemia = data_dict.get("aanemia")
            print("age",age)

            # Validate data
            if(age<0 and blood_pressure<0 and specific_gravity<0 and albumin<0 and sugar<0 and red_blood_cells<0 and pus_cell<0 and
              pus_cell_clumps<0 and bacteria<0 and blood_glucose_random<0 and blood_urea<0 and serum_creatinine<0 and sodium<0 and
              potassium<0 and haemoglobin<0 and packed_cell_volume<0 and white_blood_cell_count<0 and red_blood_cell_count<0 and
              hypertension<0 and diabetes_mellitus<0 and coronary_artery_disease<0 and appetite<0 and peda_edema<0 and aanemia<0):
                return -1

            # Preprocess data
            data = [[
                age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell,
                pus_cell_clumps, bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium,
                potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, red_blood_cell_count,
                hypertension, diabetes_mellitus, coronary_artery_disease, appetite, peda_edema, aanemia
            ]]
            df = self.preprocessData(data)

            # load model from pickle file
            model_pkl_file = "./savedModels/ckd-rf.pkl"
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
        df = pd.DataFrame(data, columns=[
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia'
        ])

        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        num_cols = [col for col in df.columns if df[col].dtype != 'object']


        # Label Encode Categorical Columns
        le = LabelEncoder()
        for c_col in cat_cols:
            df[c_col] = le.fit_transform(df[c_col])

        # scaler = StandardScaler()
        # for n_col in num_cols:
        #     scaler.fit_transform(df[n_col])

        return df
