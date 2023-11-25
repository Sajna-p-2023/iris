from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
pkl_model = pickle.load(open('savedmodel.pickle', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    result = pkl_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return render_template('result.html', **locals())

if __name__== '__main__':
    app.run(debug=True)