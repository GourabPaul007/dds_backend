from flask import Flask, request, make_response, Response, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/diabetes/", methods=['POST'])
def diabetesCheck():
    age = request.args.get("age")
    hypertension = request.args.get("hypertension")
    heart_disease = request.args.get("heart_disease")
    bmi = request.args.get("bmi")
    HbA1c_level = request.args.get("HbA1c_level")
    blood_glucose_level = request.args.get("blood_glucose_level")

    print(type(request.args))
    # d = params
    print("age",age)
    diabetesResult = modelCheck(age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level)
    # return f"params - {r}"
    return jsonify("diabetes: ", diabetesResult)


def modelCheck(age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level):
    # age                    0.026310
    # hypertension           3.545255
    # heart_disease         -0.241971
    # bmi                    1.600112
    # HbA1c_level           -0.207389
    # blood_glucose_level    2.038460
    
    # data = [[0.026310,3.545255,-0.241971,1.600112,-0.207389,2.038460]]
    # data = [[28.0,0,0,27.32,5.7,158]]
    data = [[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]]
    scaler = StandardScaler()
    scaler.fit_transform(data)
    df = pd.DataFrame(data, columns=['age', 'hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level'])
    print(df)

    model_pkl_file = "./savedModels/diabetes-model.pkl"
    # load model from pickle file
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
    # evaluate model 
    y_predict = model.predict(df)
    print(y_predict)


if __name__ == "__main__":
    app.run(debug=True)