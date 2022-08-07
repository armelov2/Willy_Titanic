import pandas as pd
import joblib
from flask import Flask, request
pipeline=joblib.load("titan.model")
import flask
print(flask.__version__)

app=Flask("__name__")

# demarrage de mon application
@app.route("/")  # création du lien

# page d'acceuil
def index():
  return "<h1> voici ma page de prédiction pour le titanic </h1>"

@app.route('/ping', methods=['GET'])
def ping():
  return ('pong', 200)


@app.route('/predict',methods=['POST'])
def predict():
  df=pd.DataFrame(request.json)  
  resultat=pipeline.predict(df)[0]
  return (str(resultat),200)

  # obligataoire pour demarrer ma page
if __name__=="__main__":
    app.run(host="0.0.0.0")
