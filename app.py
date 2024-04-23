import numpy as np
from services.brainTumorService import BrainTumorService
from services.liverService import LiverService
from services.diabetesService import DiabetesService
from services.kidneyService import KidneyService
from flask import Flask, request, make_response, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/diabetes/", methods=['POST'])
def diabetesCheck():
    data = request.args
    d = DiabetesService()
    diabetesResult = d.checkDiabetes(data)
    print(diabetesResult)
    # diabetesResult = modelCheck(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    # return f"params - {r}"
    return jsonify({'diabetes': diabetesResult})

@app.route("/liver/", methods=['POST'])
def liverCheck():
    args = request.args
    print(args)
    d = LiverService()
    liverResult = d.checkLiver(args)
    print(liverResult)
    return jsonify({'liver': liverResult})


@app.route("/kidney/", methods=['POST'])
def kidneyCheck():
    args = request.args
    print(args)
    d = KidneyService()
    kidneyResult = d.checkKidney(args)
    print(kidneyResult)
    return jsonify({'kidney': kidneyResult})

@app.route("/braintumor/", methods=['POST'])
def braintumorCheck():
    # for postman
    data = request.files.get("img")
    bts = BrainTumorService()
    btResult = bts.checkBrainTumor(data)
    return jsonify({'brain_tumor': btResult})

if __name__ == "__main__":
    app.run(debug=True)