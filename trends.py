import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def mainF():
    # Load the dataset
    df = pd.read_csv('tweets_large_data.csv')

    # Parse the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Check the date range of the dataset
    date_min = df['timestamp'].min()
    date_max = df['timestamp'].max()
    total_days = (date_max - date_min).days

    print(f"Date range: {date_min} to {date_max} (Total days: {total_days})")

    # Create a DataFrame suitable for Prophet
    df_prophet = df[['timestamp', 'like_count']].rename(columns={'timestamp': 'ds', 'like_count': 'y'})

    # Initialize the Prophet model
    model = Prophet()

    # Fit the model
    model.fit(df_prophet)

    # Create a future dataframe for predictions
    future = model.make_future_dataframe(periods=30)  # Predicting 30 days into the future

    # Make predictions
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)
    plt.show()

    # Evaluate the model (optional, using cross-validation)

    # Adjust the cross-validation parameters based on the dataset range
    initial_days = min(total_days // 2, 30)  # Use at most half the data or 30 days, whichever is smaller
    horizon_days = min(total_days // 4, 7)  # Use at most a quarter of the data or 7 days, whichever is smaller

    cv_results = cross_validation(model, initial=f'{initial_days} days', period='7 days',
                                  horizon=f'{horizon_days} days')
    performance = performance_metrics(cv_results)
    print(performance)

    # Making predictions for retweet_count (similar steps)
    df_prophet_retweet = df[['timestamp', 'retweet_count']].rename(columns={'timestamp': 'ds', 'retweet_count': 'y'})

    # Initialize the Prophet model for retweet_count
    model_retweet = Prophet()

    # Fit the model
    model_retweet.fit(df_prophet_retweet)

    # Create a future dataframe for predictions
    future_retweet = model_retweet.make_future_dataframe(periods=30)  # Predicting 30 days into the future

    # Make predictions
    forecast_retweet = model_retweet.predict(future_retweet)

    # Plot the forecast for retweet_count
    fig_retweet = model_retweet.plot(forecast_retweet)
    plt.show()

    fig_retweet.savefig('static/images/myFig.jpg')
