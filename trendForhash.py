from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import os

filepath = 'tweets_large_data.csv'


def main(hashtagA):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Parse the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Function to analyze a specific hashtag and make future predictions
    def analyze_and_predict_hashtag(df, hashtag, future_days=30):
        # Filter the dataset for the specified hashtag
        df_hashtag = df[df['hashtags'].str.contains(hashtag, na=False)]

        # Check if there are enough data points for the hashtag
        if df_hashtag.shape[0] < 2:
            print(f"Not enough data for hashtag: {hashtag}. Skipping...")
            return None, None, None

        # Group by date and sum the like and retweet counts
        df_hashtag_grouped = df_hashtag.groupby(df_hashtag['timestamp'].dt.date).agg(
            {'like_count': 'sum', 'retweet_count': 'sum'}).reset_index()
        df_hashtag_grouped.columns = ['date', 'total_likes', 'total_retweets']

        # Prepare data for Prophet
        df_likes = df_hashtag_grouped[['date', 'total_likes']].rename(columns={'date': 'ds', 'total_likes': 'y'})
        df_retweets = df_hashtag_grouped[['date', 'total_retweets']].rename(
            columns={'date': 'ds', 'total_retweets': 'y'})

        # Initialize and fit Prophet models
        model_likes = Prophet()
        model_retweets = Prophet()

        try:
            model_likes.fit(df_likes)
            model_retweets.fit(df_retweets)
        except ValueError as e:
            print(f"Error fitting model for hashtag {hashtag}: {e}")
            return None, None, None

        # Create future dataframes for predictions
        future_likes = model_likes.make_future_dataframe(periods=future_days)
        future_retweets = model_retweets.make_future_dataframe(periods=future_days)

        # Make predictions
        forecast_likes = model_likes.predict(future_likes)
        forecast_retweets = model_retweets.predict(future_retweets)

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(df_hashtag_grouped['date'], df_hashtag_grouped['total_likes'], marker='o', label='Actual Likes')
        plt.plot(df_hashtag_grouped['date'], df_hashtag_grouped['total_retweets'], marker='x', label='Actual Retweets')
        plt.plot(forecast_likes['ds'], forecast_likes['yhat'], linestyle='--', label='Predicted Likes')

        fig = plt.plot(forecast_retweets['ds'], forecast_retweets['yhat'], linestyle='--', label='Predicted Retweets')
        print(hashtag)
        plt.title(f'Number of Likes and Retweets for Hashtag: {hashtagA}')
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        my_file = Path("static/images/hashtag_likes_retweets.png")
        if my_file.exists():
            os.remove("static/images/hashtag_likes_retweets.png")
        plt.savefig('static/images/hashtag_likes_retweets.png')
        plt.show()

        return df_hashtag_grouped, forecast_likes, forecast_retweets

    # Specify the hashtag you want to analyze
    hashtag_to_analyze = hashtagA

    # Analyze the specified hashtag and predict future values
    results, forecast_likes, forecast_retweets = analyze_and_predict_hashtag(df, hashtag_to_analyze)

    if results is not None:
        print("Actual data:")
        print(results)
        print("\nPredicted Likes:")
        print(forecast_likes[['ds', 'yhat']].tail(10))
        print("\nPredicted Retweets:")
        print(forecast_retweets[['ds', 'yhat']].tail(10))





