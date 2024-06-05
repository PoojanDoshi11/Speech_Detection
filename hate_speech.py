# hate_speech.py

import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')

class HateSpeechDetection:
    def __init__(self, data_path):
        self.data_path = data_path
        self.stemmer = nltk.SnowballStemmer('english')
        self.stop_words_set = set(stopwords.words('english'))
        self.cv = CountVectorizer()
        self.model = DecisionTreeClassifier()
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        dataset = pd.read_csv(self.data_path)
        dataset['labels'] = dataset['class'].map({0: 'Hate Speech', 1: 'Offensive Language', 2: 'No Hate or Offensive Language'})
        self.data = dataset[['tweet', 'labels']]
    
    def clean_data(self, text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\W', '', text)
        text = ' '.join([word for word in text.split() if word not in self.stop_words_set])
        text = ' '.join([self.stemmer.stem(word) for word in text.split()])
        return text

    def preprocess_data(self):
        self.data['tweet'] = self.data['tweet'].apply(self.clean_data)
        X = np.array(self.data['tweet'])
        Y = np.array(self.data['labels'])
        x = self.cv.fit_transform(X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, Y, test_size=0.33, random_state=42)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='YlGnBu')
        # plt.show()
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')

    def predict(self, text):
        text = self.clean_data(text)
        text_vectorized = self.cv.transform([text]).toarray()
        prediction = self.model.predict(text_vectorized)
        print(prediction)
        return prediction

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()
