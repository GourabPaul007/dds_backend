import numpy as np
from services.heartDiseaseService import HeartDiseaseService
from services.lungCancerService import LungCancerService
from services.covidService import CovidService
from services.breastCancerService import BreastCancerService
from services.strokeService import StrokeService
from services.parkinsonsService import ParkinsonsService
from services.alzheimersService import AlzheimersService
from services.tuberculosisService import TuberculosisService
from services.pneumoniaService import PneumoniaService
from services.brainTumorService import BrainTumorService
from services.liverService import LiverService
from services.diabetesService import DiabetesService
from services.kidneyService import KidneyService
from flask import Flask, request, make_response, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

@app.route("/alzheimers/", methods=['POST'])
def alzheimersCheck():
    data = request.files.get("img")
    al = AlzheimersService()
    print("hello world")
    print(data)
    alResult = al.checkAlzheimers(data)
    return jsonify(alResult)

@app.route("/braintumor/", methods=['POST'])
def braintumorCheck():
    data = request.files.get("img")
    bts = BrainTumorService()
    btResult = bts.checkBrainTumor(data)
    return jsonify(btResult)

@app.route("/breastcancer/", methods=['POST'])
def breastCancerCheck():
    args = request.args
    bc = BreastCancerService()
    print(args)
    bcResult = bc.checkBreastCancer(args)
    return jsonify(bcResult)

@app.route("/covid/", methods=['POST'])
def covidCheck():
    data = request.files.get("img")
    covid = CovidService()
    covidResult = covid.checkCovid(data)
    return jsonify(covidResult)

@app.route("/diabetes/", methods=['POST'])
def diabetesCheck():
    data = request.args
    d = DiabetesService()
    diabetesResult = d.checkDiabetes(data)
    print(diabetesResult)
    # diabetesResult = modelCheck(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    # return f"params - {r}"
    return jsonify(diabetesResult)

@app.route("/heartdisease/", methods=['POST'])
def heartDiseaseCheck():
    args = request.args
    hd = HeartDiseaseService()
    print(args)
    hdResult = hd.checkHeartDisease(args)
    return jsonify(hdResult)

@app.route("/kidney/", methods=['POST'])
def kidneyCheck():
    args = request.args
    print(args)
    d = KidneyService()
    kidneyResult = d.checkKidney(args)
    print(kidneyResult)
    return jsonify(kidneyResult)

@app.route("/lungcancer/", methods=['POST'])
def lungCancerCheck():
    args = request.args
    lc = LungCancerService()
    print(args)
    lcResult = lc.checkLungCancer(args)
    return jsonify(lcResult)

@app.route("/parkinsons/", methods=['POST'])
def parkinsonsCheck():
    args = request.args
    print(args)
    d = ParkinsonsService()
    parkinsonsResult = d.checkParkinsons(args)
    print(parkinsonsResult)
    return jsonify(parkinsonsResult)

@app.route("/pneumonia/", methods=['POST'])
def pneumoniaCheck():
    # bad output
    data = request.files.get("img")
    ps = PneumoniaService()
    pResult = ps.checkPneumonia(data)
    return jsonify({'pneumonia': pResult})

@app.route("/stroke/", methods=['POST'])
def strokeCheck():
    args = request.args
    ss = StrokeService()
    print(args)
    ssResult = ss.checkStroke(args)
    return jsonify(ssResult)

@app.route("/tuberculosis/", methods=['POST'])
def tuberculosisCheck():
    data = request.files.get("img")
    tb = TuberculosisService()
    tbResult = tb.checkTuberculosis(data)
    return jsonify(tbResult)


# TODO: Lung Cancer
# TODO: Hepatitis C

if __name__ == "__main__":
    app.run(debug=True)