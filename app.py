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



if __name__ == "__main__":
    app.run(debug=True)