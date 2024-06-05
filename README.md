# Speech Type Detection

This project utilizes a machine learning model to detect hate speech and offensive language in social media posts. Implemented using Python and Flask, it provides a web interface for users to input text and receive predictions. The model classifies text into three categories: "Hate Speech," "Offensive Language," and "No Hate or Offensive Language," achieving an accuracy of 87.3%.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About

This project focuses on detecting hate speech and offensive language in social media posts. Using a machine learning model trained on a labeled dataset, the application can classify text into three categories: "Hate Speech," "Offensive Language," and "No Hate or Offensive Language." The project is implemented using Python and Flask, providing a web interface to input text and get predictions.

## Features

- **Data Preprocessing:** 
  - **Text Cleaning:** Removal of URLs, HTML tags, punctuation, digits, and stopwords. 
  - **Stemming:** Reducing words to their root form using NLTK's Snowball Stemmer.

- **Model Training:**
  - **Algorithm:** Decision Tree Classifier for classification.
  - **Vectorization:** Text data is converted into numerical features using CountVectorizer.

- **Web Interface:**
  - **Flask Framework:** Simple web application to input text and display predictions.
  - **Endpoints:** 
    - `/`: Renders the home page.
    - `/predict`: Accepts text input via POST method and returns the prediction result.

- **Evaluation:**
  - **Metrics:** Confusion matrix and accuracy score to evaluate model performance.
  - **Visualization:** Heatmap of the confusion matrix using Seaborn.

- **Easy to Use:**
  - **Beginner Friendly:** Clear structure and detailed code comments for easy understanding.
  - **Deployment Ready:** Flask app setup for local deployment.

## Installation

Provide instructions on how to install and set up your project locally. Include any dependencies that need to be installed and any configuration steps.

```bash
# Clone the repository
git clone https://github.com/PoojanDoshi11/Speech_Detection.git

# Navigate to the project directory
cd Speech_Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage
-- Run the Flask app
```bash
python app.py
```
-- then follow the provided local host link
-- then enter some text and click on detect
-- then enjoy the results
