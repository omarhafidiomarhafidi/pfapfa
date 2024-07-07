import joblib
from flask import Flask, render_template, request, jsonify, make_response, send_from_directory
from trends import mainF
from trendForhash import main

app = Flask(__name__)
target_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

clf = joblib.load('emotion_classifier.joblib')

# Load the saved vectorizer if needed
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.after_request
def add_header(response):
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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

    return render_template('trends.html')

@app.route('/static/images/<path:filename>')
def custom_static(filename):
    response = make_response(send_from_directory('static/images', filename))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/trends', methods=['POST'])
def trendsp():
    text = request.form['text']
    print("printing text : "+text)
    main(text)
    return render_template("trends.html",isSecond=True)



if __name__ == '__main__':
    app.run(debug=True)
