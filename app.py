from flask import Flask, render_template, request


import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_class():
    Sepal_Length=float(request.form.get("Sepal Length(cm)"))
    Sepal_Width=float(request.form.get("Sepal Width(cm)"))
    Petal_Length=float(request.form.get("Petal Length(cm)"))
    Petal_Width=float(request.form.get("Petal Width(cm)"))

    #predict model

    result = model.predict(np.array([Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]).reshape(1,4))
    
    if result[0]== "Iris-setosa":
        result = "Iris-setosa"
    elif result[0]=="Iris-versicolor":
        result = "Iris-versicolor"
    else:
        result = "Iris-virginica"

    
    return f"The class of this iris is {str(result)}"

if __name__ == "__main__":
    app.run(debug=True)