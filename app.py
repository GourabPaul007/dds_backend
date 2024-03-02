from services.liverService import LiverService
from services.diabetesService import DiabetesService
from flask import Flask, request, make_response, Response, jsonify

app = Flask(__name__)

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
    data = request.args
    d = LiverService()
    liverResult = d.checkLiver(data)
    print(liverResult)
    return jsonify({'liver': liverResult})


if __name__ == "__main__":
    app.run(debug=True)