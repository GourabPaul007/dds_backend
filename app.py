import numpy as np
from services.breastCancer import BreastCancerService
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
    return jsonify(diabetesResult)

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
    return jsonify(kidneyResult)

@app.route("/parkinsons/", methods=['POST'])
def parkinsonsCheck():
    args = request.args
    print(args)
    d = ParkinsonsService()
    parkinsonsResult = d.checkParkinsons(args)
    print(parkinsonsResult)
    return jsonify(parkinsonsResult)

@app.route("/braintumor/", methods=['POST'])
def braintumorCheck():
    data = request.files.get("img")
    bts = BrainTumorService()
    btResult = bts.checkBrainTumor(data)
    return jsonify({'brain_tumor': btResult})

@app.route("/pneumonia/", methods=['POST'])
def pneumoniaCheck():
    # bad output
    data = request.files.get("img")
    ps = PneumoniaService()
    pResult = ps.checkPneumonia(data)
    return jsonify({'pneumonia': pResult})

@app.route("/tuberculosis/", methods=['POST'])
def tuberculosisCheck():
    data = request.files.get("img")
    tb = TuberculosisService()
    tbResult = tb.checkTuberculosis(data)
    return jsonify(tbResult)

@app.route("/alzheimers/", methods=['POST'])
def alzheimersCheck():
    data = request.files.get("img")
    al = AlzheimersService()
    print("hello world")
    print(data)
    alResult = al.checkAlzheimers(data)
    return jsonify(alResult)

@app.route("/breast_cancer/", methods=['POST'])
def breastCancerCheck():
    args = request.args
    bc = BreastCancerService()
    print(args)
    bcResult = bc.checkBreastCancer(args)
    return jsonify(bcResult)

# TODO: Lung Cancer
# TODO: Hepatitis C
# TODO: Stroke

if __name__ == "__main__":
    app.run(debug=True)