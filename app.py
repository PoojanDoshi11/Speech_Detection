# app.py

from flask import Flask, request, render_template
from hate_speech import HateSpeechDetection

app = Flask(__name__)

# Initialize the hate speech detection model globally
hate_speech_detector = HateSpeechDetection(data_path='hate_speech.csv')
hate_speech_detector.run()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = hate_speech_detector.predict(text)[0]
    if prediction == 'Offensive Language':
        result = 'Offensive Language'
    elif prediction == 'Hate Speech':
        result = 'Hate Speech'
    else:
        result = 'Neither having Hate nor to be Offensive'
    return render_template('result.html', text=text, result=result)

if __name__ == '__main__':
    app.run(debug=False, port=8002)
