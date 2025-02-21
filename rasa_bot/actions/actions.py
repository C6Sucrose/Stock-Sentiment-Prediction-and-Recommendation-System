# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


# rasa_bot/actions/actions.py

# rasa_bot/actions/actions.py

# rasa_bot/actions/actions.py

# rasa_bot/actions/actions.py

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from functools import lru_cache
import time
import logging
from rasa_sdk.events import SlotSet
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import joblib
import os



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CACHE_TIMEOUT = 3600  # 1 hour cache timeout
RATE_LIMIT_CALLS = 100  # Maximum calls per minute
RATE_LIMIT_PERIOD = 60  # Time period in seconds
VALID_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']


class ValidateStockSlot(Action):
    def name(self) -> Text:
        return "action_validate_stock"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        stock = tracker.get_slot('stock')
        valid_stocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']

        if stock and stock.upper() in valid_stocks:
            return [SlotSet("stock", stock.upper())]
        else:
            dispatcher.utter_message(response="utter_invalid_stock")
            return [SlotSet("stock", None)]

class RateLimiter:
    def __init__(self, max_calls, period):
        self.calls = []
        self.max_calls = max_calls
        self.period = period

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

# Data loading utilities with correct function signatures
@lru_cache(maxsize=1)
def load_models(timeout=CACHE_TIMEOUT):
    """Load models with caching"""
    try:
        models_dir = os.path.abspath('../models')
        reg_model_path = os.path.join(models_dir, 'random_forest_regressor.joblib')
        scaler_path = os.path.join(models_dir, 'scaler.joblib')

        if not os.path.exists(reg_model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Required model files not found")

        reg_model = joblib.load(reg_model_path)
        scaler = joblib.load(scaler_path)
        return reg_model, scaler
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@lru_cache(maxsize=1)
def load_stock_data(timeout=CACHE_TIMEOUT):
    """Load and cache stock data"""
    try:
        data_path = os.path.abspath('../data/processed/cleaned_stock_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError("Stock data file not found")
        return pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        raise

@lru_cache(maxsize=1)
def load_sentiment_data(timeout=CACHE_TIMEOUT):
    """Load and cache sentiment data"""
    try:
        sentiment_path = os.path.abspath('../data/raw/reddit_sentiment.csv')
        if not os.path.exists(sentiment_path):
            raise FileNotFoundError("Sentiment data file not found")
        return pd.read_csv(sentiment_path)
    except Exception as e:
        logger.error(f"Error loading sentiment data: {e}")
        raise

def validate_stock(stock: str) -> bool:
    """Validate if the stock symbol is in our allowed list"""
    return stock.upper() in VALID_STOCKS if stock else False


class ActionRecommendStock(Action):
    def name(self) -> Text:
        return "action_recommend_stock"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        try:
            # Load the latest data
            data_path = os.path.abspath('../data/processed/augmented_stock_data.csv')
            df = pd.read_csv(data_path)

            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Get the latest data for each stock
            latest_data = df.sort_values('Date').groupby('Ticker').last()

            # Load models
            models_dir = os.path.abspath('../models')
            rf_reg = joblib.load(os.path.join(models_dir, 'random_forest_regressor.joblib'))
            rf_clf = joblib.load(os.path.join(models_dir, 'random_forest_classifier.joblib'))

            # Prepare features
            features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Avg_Sentiment_Score']

            recommendations = []
            for ticker in latest_data.index:
                X = latest_data.loc[[ticker], features]

                # Make predictions
                price_pred = rf_reg.predict(X)[0]
                trend_prob = rf_clf.predict_proba(X)[0][1]  # Probability of increase
                sentiment_score = latest_data.loc[ticker, 'Avg_Sentiment_Score']

                # Calculate recommendation score
                rec_score = (0.4 * trend_prob) + (0.3 * (sentiment_score + 1) / 2) + (
                            0.3 * (price_pred - latest_data.loc[ticker, 'Close']) / latest_data.loc[ticker, 'Close'])

                recommendations.append((ticker, rec_score))

            # Sort recommendations by score
            recommendations.sort(key=lambda x: x[1], reverse=True)

            # Format the response
            response = "**Stock Recommendations**\n"
            response += "Here are my top recommendations based on technical analysis, sentiment, and predicted growth:\n"

            for ticker, score in recommendations:
                sentiment = latest_data.loc[ticker, 'Avg_Sentiment_Score']
                sentiment_emoji = "📈" if sentiment > 0 else "📉" if sentiment < 0 else "⚖️"
                response += f"**{ticker}** {sentiment_emoji}\n"
                response += f"Recommendation Score: {score:.2f}\n"
                response += f"Current Price: ${latest_data.loc[ticker, 'Close']:.2f}\n"

            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text=f"Sorry, I encountered an error while generating recommendations: {str(e)}")

        return []


class ActionGetSentiment(Action):
    def name(self) -> Text:
        return "action_get_sentiment"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        try:
            # Get the stock from slot
            stock = tracker.get_slot('stock')
            if not stock:
                dispatcher.utter_message(text="I need a stock symbol to analyze sentiment.")
                return []

            # Load the latest data
            data_path = os.path.abspath('../data/processed/augmented_stock_data.csv')
            df = pd.read_csv(data_path)

            # Filter for the specific stock
            stock_data = df[df['Ticker'] == stock].sort_values('Date').tail(1)

            if stock_data.empty:
                dispatcher.utter_message(text=f"Sorry, I couldn't find sentiment data for {stock}.")
                return []

            # Get sentiment score
            sentiment_score = stock_data['Avg_Sentiment_Score'].iloc[0]

            # Determine sentiment category and emoji
            if sentiment_score > 0.3:
                sentiment = "Positive 📈"
            elif sentiment_score < -0.3:
                sentiment = "Negative 📉"
            else:
                sentiment = "Neutral ⚖️"

            # Format the response
            response = (f"**{stock} Sentiment Analysis**\n"
                        f"Current Sentiment: {sentiment}\n"
                        f"Sentiment Score: {sentiment_score:.2f}")

            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text=f"Sorry, I encountered an error while analyzing sentiment: {str(e)}")

        return []


class ActionInquireStockDetails(Action):
    def name(self) -> Text:
        return "action_inquire_stock_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        try:
            # Get the stock from slot
            stock = tracker.get_slot('stock')
            if not stock:
                dispatcher.utter_message(text="I need a stock symbol to show details.")
                return []

            # Load the latest data
            data_path = os.path.abspath('../data/processed/augmented_stock_data.csv')
            df = pd.read_csv(data_path)

            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Get the latest data for the stock
            stock_data = df[df['Ticker'] == stock].sort_values('Date').tail(1)

            if stock_data.empty:
                dispatcher.utter_message(text=f"Sorry, I couldn't find details for {stock}.")
                return []

            # Calculate price change
            prev_data = df[df['Ticker'] == stock].sort_values('Date').tail(2)
            if len(prev_data) > 1:
                price_change = ((stock_data['Close'].iloc[0] - prev_data['Close'].iloc[0]) / prev_data['Close'].iloc[
                    0]) * 100
                change_emoji = "📈" if price_change > 0 else "📉"
            else:
                price_change = 0
                change_emoji = "⚖️"

            # Format the response
            response = (f"**{stock} Stock Details** {change_emoji}\n"
                        f"Last Trading Date: {stock_data['Date'].iloc[0].strftime('%Y-%m-%d')}.\n"
                        f"Current Price: ${stock_data['Close'].iloc[0]:.2f}.\n"
                        f"Day's Range: ${stock_data['Low'].iloc[0]:.2f} - ${stock_data['High'].iloc[0]:.2f}.\n"
                        f"Trading Volume: {stock_data['Volume'].iloc[0]:,.0f}.\n"
                        f"Price Change: {price_change:+.2f}%.\n"
                        f"Sentiment Score: {stock_data['Avg_Sentiment_Score'].iloc[0]:.2f}.")

            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text=f"Sorry, I encountered an error while retrieving stock details: {str(e)}")

        return []


class ActionPredictStock(Action):
    def name(self) -> Text:
        return "action_predict_stock"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[
        Dict[Text, Any]]:
        try:
            # Get the stock from slot
            stock = tracker.get_slot('stock')
            if not stock:
                dispatcher.utter_message(text="I need a stock symbol to make a prediction.")
                return []

            # Load the latest data
            data_path = os.path.abspath('../data/processed/augmented_stock_data.csv')
            df = pd.read_csv(data_path)

            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Filter for the specific stock
            stock_data = df[df['Ticker'] == stock].sort_values('Date').tail(1)

            if stock_data.empty:
                dispatcher.utter_message(text=f"Sorry, I couldn't find data for {stock}.")
                return []

            # Load models and scalers
            models_dir = os.path.abspath('../models')
            rf_reg = joblib.load(os.path.join(models_dir, 'random_forest_regressor.joblib'))
            rf_clf = joblib.load(os.path.join(models_dir, 'random_forest_classifier.joblib'))

            # Prepare features
            features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Avg_Sentiment_Score']
            X = stock_data[features]

            # Make predictions
            price_pred = rf_reg.predict(X)[0]
            trend_prob = rf_clf.predict_proba(X)[0]

            # Get current price
            current_price = stock_data['Close'].iloc[0]

            # Format the response
            trend = "increase" if trend_prob[1] > 0.5 else "decrease"
            confidence = max(trend_prob) * 100

            response = (f"**{stock} Stock Prediction**.\n"
                        f"Current Price: ${current_price:.2f}.\n"
                        f"Predicted Price: ${price_pred:.2f}.\n"
                        f"Trend Prediction: {trend.title()} (Confidence: {confidence:.1f}%).")

            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text=f"Sorry, I encountered an error while making the prediction: {str(e)}")

        return []