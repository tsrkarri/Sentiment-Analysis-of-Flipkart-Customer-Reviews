from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def pred():
    review = request.form.get('review')
    data_point = [review]

    model = joblib.load('best_models/svc.pkl')

    pred = model.predict(data_point)

    if pred == [0]:
        sentiment = 'Sentiment is Negative'
    else :
        sentiment = 'Sentiment is Positive'

    return render_template('index.html', sentiment = sentiment)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')