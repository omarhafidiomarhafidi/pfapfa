import joblib
from flask import Flask, render_template, request, jsonify
from trends import mainF
app = Flask(__name__)
target_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

clf = joblib.load('emotion_classifier.joblib')

# Load the saved vectorizer if needed
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Function to preprocess and predict a single tweet
def analyse_tweet(tweet):
  # Vectorize the tweet using the loaded vectorizer
  tweet_tfidf = vectorizer.transform([tweet])
  # Predict the sentiment of the new tweet using the loaded model
  prediction = clf.predict(tweet_tfidf)
  predicted_label = target_names[prediction[0]]
  return predicted_label

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    predicted_sentiment = analyse_tweet(text)
    response = jsonify({'result': predicted_sentiment, 'index': target_names.index(predicted_sentiment)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/trends', methods=['GET'])
def trends():
    mainF()
    return render_template('trends.html')
@app.route('/trends', methods=['POST'])
def trendsp():
    data = request.get_json()
    text= data.get('text', '')
    return render_template('trends.html')


if __name__ == '__main__':
    app.run()
