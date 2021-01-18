from flask import Flask,jsonify,request
from test import GetPrediction
app = Flask(__name__)

@app.route('/Predict-alphabet',methods= ['POST'])

def PredictData():
    Image = request.files.get('alphabet')
    Prediction = GetPrediction(Image)
    return jsonify({
        'Prediction': Prediction
    }),200   

if (__name__ == '__main__'):
    app.run(debug = True)