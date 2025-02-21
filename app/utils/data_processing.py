# app/utils/data_processing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from faker import Faker
from scipy import stats
import logging
import yfinance as yf
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from dotenv import load_dotenv  # To load environment variables
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Faker for fake data generation
fake = Faker()
Faker.seed(0)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def is_file_fresh(file_path, hours=24):
    """
    Check if a file is fresh (modified within the last 'hours' hours).

    Parameters:
    - file_path (str): Path to the file.
    - hours (int): Freshness threshold in hours.

    Returns:
    - bool: True if file is fresh, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    return datetime.now() - file_mtime < timedelta(hours=hours)


def load_and_prepare_data(fetch=True, manual_refresh=False):
    """
    Load, clean, transform, and augment stock and sentiment data with added freshness check.

    Parameters:
    - fetch (bool): Whether to fetch new data.
    - manual_refresh (bool): Indicates if the data refresh was manually triggered.

    Returns:
    - cleaned_stock_data_scaled (DataFrame): The cleaned and scaled real stock data.
    - augmented_data_scaled (DataFrame): The cleaned and augmented (real + fake) scaled stock data.
    - sentiment_data (DataFrame): The sentiment data with scores.
    - cleaned_stock_data_unscaled (DataFrame): The cleaned real stock data before scaling.
    - augmented_data_unscaled (DataFrame): The cleaned and augmented (real + fake) stock data before scaling.
    """
    # Define paths
    raw_stock_data_path = os.path.abspath('../data/raw/stock_data.csv')
    processed_stock_data_path = os.path.abspath('../data/processed/cleaned_stock_data.csv')
    sentiment_data_path = os.path.abspath('../data/raw/reddit_sentiment.csv')
    augmented_stock_data_path = os.path.abspath('../data/processed/augmented_stock_data.csv')

    # Determine if data needs to be fetched
    fetch_data = fetch and manual_refresh
    data_fresh = all([
        os.path.exists(raw_stock_data_path),
        os.path.exists(processed_stock_data_path),
        os.path.exists(sentiment_data_path),
        os.path.exists(augmented_stock_data_path),
        is_file_fresh(raw_stock_data_path),
        is_file_fresh(processed_stock_data_path),
        is_file_fresh(sentiment_data_path),
        is_file_fresh(augmented_stock_data_path)
    ])

    if not data_fresh and fetch:
        logger.info("Data is not fresh or manually refreshed. Proceeding to fetch new data.")
        logger.info("Fetching stock data...")
        fetch_stock_data(raw_stock_data_path)
        logger.info("Fetching sentiment data...")
        fetch_sentiment_data(sentiment_data_path)
    else:
        logger.info("Using existing data without fetching new data.")

    # Load and clean stock data
    try:
        cleaned_stock_data_scaled, cleaned_stock_data_unscaled = clean_stock_data(
            raw_path=raw_stock_data_path,
            processed_path=processed_stock_data_path,
            columns_to_clean=['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        )
    except Exception as e:
        logger.error(f"Failed to clean stock data: {e}")
        raise

    # Feature scaling
    try:
        scaler = StandardScaler()
        numerical_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        if not all(feature in cleaned_stock_data_scaled.columns for feature in numerical_features):
            missing = list(set(numerical_features) - set(cleaned_stock_data_scaled.columns))
            logger.error(f"Missing numerical features for scaling: {missing}")
            raise KeyError(f"Missing numerical features: {missing}")
        cleaned_stock_data_scaled[numerical_features] = scaler.fit_transform(cleaned_stock_data_scaled[numerical_features])
        logger.info("Feature scaling applied to numerical columns.")
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise

    # Augment data with fake data
    try:
        augmented_data_scaled, augmented_data_unscaled = augment_with_fake_data(cleaned_stock_data_unscaled, percentage=0.5)
        augmented_data_scaled.to_csv(augmented_stock_data_path, index=False)
        logger.info(f"Augmented stock data saved to '{augmented_stock_data_path}'.")
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        raise

    # Load sentiment data
    try:
        sentiment_data = pd.read_csv(sentiment_data_path)
        logger.info(f"Loaded sentiment data from '{sentiment_data_path}'.")
    except Exception as e:
        logger.error(f"Failed to load sentiment data: {e}")
        raise

    # Ensure 'Sentiment' column exists
    if 'Sentiment' in sentiment_data.columns:
        try:
            sentiment_data['score'] = sentiment_data['Sentiment'].apply(calculate_sentiment_score)
            logger.info("Calculated 'score' column for sentiment data.")
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {e}")
            sentiment_data['score'] = 0  # Assign default value
            logger.warning("Assigned default 'score' value of 0 due to errors.")
    else:
        logger.warning("Sentiment data does not contain 'Sentiment' column. Assigning default 'score' value of 0.")
        sentiment_data['score'] = 0  # Assign default value

    # Validate sentiment_data
    if sentiment_data['score'].isnull().all():
        logger.error("All sentiment scores are NaN. Check sentiment data processing.")
        raise ValueError("Sentiment scores could not be calculated.")

    return cleaned_stock_data_scaled, augmented_data_scaled, sentiment_data, cleaned_stock_data_unscaled, augmented_data_unscaled


def fetch_stock_data(raw_stock_data_path):
    """
    Fetch historical stock data for predefined tickers and save to CSV.

    Parameters:
    - raw_stock_data_path (str): Path to save the raw stock data CSV.
    """
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    end_date = datetime.today()
    start_date = end_date - timedelta(days=1825)  # Approximately 5 years

    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        stock_entries = []
        for ticker in tickers:
            if ticker not in data.columns.levels[0]:
                logger.warning(f"No data found for ticker '{ticker}'. Skipping.")
                continue
            ticker_data = data[ticker].reset_index()
            ticker_data['Ticker'] = ticker
            # Ensure 'Dividends' and 'Stock Splits' are present
            for col in ['Dividends', 'Stock Splits']:
                if col not in ticker_data.columns:
                    # Attempt to fetch dividends and stock splits separately
                    # If unavailable, assign realistic random values
                    if col == 'Dividends':
                        ticker_data[col] = np.random.uniform(0.0, 1.0, size=len(ticker_data))
                        logger.info(f"Assigned random 'Dividends' for '{ticker}'.")
                    elif col == 'Stock Splits':
                        ticker_data[col] = np.random.uniform(0.0, 10.0, size=len(ticker_data))  # e.g., 2.0 for a 2-for-1 split
                        logger.info(f"Assigned random 'Stock Splits' for '{ticker}'.")
                else:
                    # Replace NaNs with 0.0 and add small random noise for realism
                    ticker_data[col].fillna(0.0, inplace=True)
                    ticker_data[col] = ticker_data[col] + np.random.uniform(-0.01, 0.01, size=len(ticker_data))
            stock_entries.append(ticker_data)
        if stock_entries:
            stock_df = pd.concat(stock_entries, ignore_index=True)
            stock_df.to_csv(raw_stock_data_path, index=False)
            logger.info(f"Stock data fetched and saved to '{raw_stock_data_path}'.")
        else:
            logger.warning("No stock data fetched. Saving empty DataFrame.")
            pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Ticker']).to_csv(raw_stock_data_path, index=False)
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise

def fetch_sentiment_data(sentiment_data_path):
    """
    Fetch recent Reddit posts from r/wallstreetbets for each ticker and perform sentiment analysis.

    Parameters:
    - sentiment_data_path (str): Path to save the sentiment data CSV.
    """
    # Reddit API credentials should be set as environment variables for security
    reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
    reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    reddit_user_agent = os.getenv('REDDIT_USER_AGENT')

    if not all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
        logger.error("Reddit API credentials are not set. Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT environment variables.")
        raise EnvironmentError("Reddit API credentials not set.")

    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']
    limit_per_ticker = 1000

    try:
        reddit = praw.Reddit(client_id=reddit_client_id,
                             client_secret=reddit_client_secret,
                             user_agent=reddit_user_agent)
    except Exception as e:
        logger.error(f"Error initializing Reddit instance: {e}")
        raise

    sentiment_entries = []

    for ticker in tickers:
        try:
            subreddit = reddit.subreddit('wallstreetbets')
            posts = subreddit.search(ticker, limit=limit_per_ticker)
            for post in posts:
                title = post.title
                sentiment = analyze_sentiment(title)
                sentiment_entries.append({
                    'Ticker': ticker,
                    'Title': title,
                    'Sentiment': sentiment
                })
            logger.info(f"Fetched and analyzed {limit_per_ticker} posts for '{ticker}'.")
        except Exception as e:
            logger.error(f"Error fetching/analyzing posts for '{ticker}': {e}")
            continue  # Proceed to next ticker if error occurs

    # Ensure there is data to save
    if sentiment_entries:
        sentiment_df = pd.DataFrame(sentiment_entries)
        sentiment_df.to_csv(sentiment_data_path, index=False)
        logger.info(f"Sentiment data saved to '{sentiment_data_path}'.")
    else:
        logger.warning("No sentiment data fetched. Saving empty DataFrame.")
        pd.DataFrame(columns=['Ticker', 'Title', 'Sentiment']).to_csv(sentiment_data_path, index=False)

def analyze_sentiment(text):
    """
    Analyze sentiment of the given text using VADER.

    Parameters:
    - text (str): Text to analyze.

    Returns:
    - sentiment (str): 'positive', 'neutral', or 'negative' based on score.
    """
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def clean_stock_data(raw_path, processed_path, columns_to_clean):
    """
    Clean stock data by capping outliers using the IQR method.

    Parameters:
    - raw_path (str): Path to the raw stock data CSV.
    - processed_path (str): Path to save the cleaned stock data CSV.
    - columns_to_clean (list): Columns to apply outlier capping.

    Returns:
    - df_scaled (DataFrame): The cleaned and scaled stock data.
    - df_unscaled (DataFrame): The cleaned stock data before scaling.
    """
    try:
        df = pd.read_csv(raw_path)
        logger.info(f"Loaded raw stock data from '{raw_path}'.")
    except Exception as e:
        logger.error(f"Failed to read raw stock data: {e}")
        raise

    # Ensure 'Date' is in datetime format and sort
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        logger.info("Processed 'Date' column to datetime and sorted data.")
    except Exception as e:
        logger.error(f"Error processing 'Date' column: {e}")
        raise

    # Cap outliers using the IQR method
    try:
        for column in columns_to_clean:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in stock data. Skipping outlier capping for this column.")
                continue
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            original_min = df[column].min()
            original_max = df[column].max()
            df[column] = np.where(df[column] < lower_bound, lower_bound,
                                  np.where(df[column] > upper_bound, upper_bound, df[column]))
            logger.info(f"Capped '{column}' column. Original min: {original_min}, Original max: {original_max}. New bounds: {lower_bound} - {upper_bound}.")
        logger.info("Outliers capped using the IQR method.")
    except Exception as e:
        logger.error(f"Error during outlier capping: {e}")
        raise

    # Save the cleaned data before scaling (unscaled)
    try:
        df_unscaled = df.copy()
        logger.info("Created unscaled copy of cleaned stock data.")
    except Exception as e:
        logger.error(f"Error creating unscaled copy of cleaned stock data: {e}")
        raise

    # Feature scaling
    try:
        scaler = StandardScaler()
        numerical_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        if not all(feature in df.columns for feature in numerical_features):
            missing = list(set(numerical_features) - set(df.columns))
            logger.error(f"Missing numerical features for scaling: {missing}")
            raise KeyError(f"Missing numerical features: {missing}")
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        logger.info("Feature scaling applied to numerical columns.")
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise

    # Save the cleaned and scaled data
    try:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory for processed data: {e}")
        raise

    try:
        df.to_csv(processed_path, index=False)
        logger.info(f"Cleaned and scaled stock data saved to '{processed_path}'.")
    except PermissionError:
        logger.error(f"Permission denied: Unable to write to '{processed_path}'. Please close the file if it's open and check write permissions.")
        raise
    except Exception as e:
        logger.error(f"Failed to save cleaned and scaled stock data: {e}")
        raise
    # Save the scaler for later inverse transformation
    scaler_path = os.path.abspath('../models/scaler.joblib')
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to '{scaler_path}'.")
    except Exception as e:
        logger.error(f"Failed to save scaler: {e}")
        raise

    return df, df_unscaled




def augment_with_fake_data(cleaned_stock_data_unscaled, percentage=0.5):
    """
    Augment real stock data with fake data.

    Parameters:
    - cleaned_stock_data_unscaled (DataFrame): The cleaned real stock data before scaling.
    - percentage (float): Percentage of fake data to generate relative to real data.

    Returns:
    - augmented_data_scaled (DataFrame): The augmented stock data (real + fake) with scaling applied.
    - augmented_data_unscaled (DataFrame): The augmented stock data (real + fake) before scaling.
    """
    if 'Avg_Sentiment_Score' not in cleaned_stock_data_unscaled.columns:
        # Generate random sentiment scores for existing data if missing
        cleaned_stock_data_unscaled['Avg_Sentiment_Score'] = np.random.uniform(-1.0, 1.0, size=len(cleaned_stock_data_unscaled))
        logger.info("Added missing Avg_Sentiment_Score column to existing data")

    fake_df_unscaled = generate_fake_data(cleaned_stock_data_unscaled, percentage)
    augmented_data_unscaled = pd.concat([cleaned_stock_data_unscaled, fake_df_unscaled], ignore_index=True)
    logger.info(f"Augmented unscaled data now contains {len(augmented_data_unscaled)} entries.")

    # Apply scaling to augmented data
    scaler = StandardScaler()
    numerical_features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Avg_Sentiment_Score']
    if not all(feature in augmented_data_unscaled.columns for feature in numerical_features):
        missing = list(set(numerical_features) - set(augmented_data_unscaled.columns))
        logger.error(f"Missing numerical features for scaling augmented data: {missing}")
        raise KeyError(f"Missing numerical features: {missing}")
    augmented_data_scaled = augmented_data_unscaled.copy()
    augmented_data_scaled[numerical_features] = scaler.fit_transform(augmented_data_scaled[numerical_features])
    logger.info("Feature scaling applied to augmented numerical columns.")

    # Save the scaler for 'Close' price
    # Create a separate scaler for Close column
    augmented_data_scaled_close = augmented_data_scaled.copy()
    close_scaler = StandardScaler()
    augmented_data_scaled_close['Close'] = close_scaler.fit_transform(augmented_data_unscaled[['Close']])

    # Save models to disk
    models_dir = os.path.abspath('../models')
    os.makedirs(models_dir, exist_ok=True)

    logger.info("Close Values before Scaler:")
    logger.info(augmented_data_unscaled["Close"])
    logger.info("Close Values after Scaler:")
    logger.info(augmented_data_scaled_close["Close"])
    joblib.dump(close_scaler, os.path.join(models_dir, 'scaler_close.joblib'))
    logger.info(f"'Close' scaler saved to '{models_dir}/scaler_close.joblib'.")

    # Save the scaler for input features
    logger.info("Close Values before Scaler:")
    logger.info(augmented_data_unscaled[numerical_features])
    logger.info("Close Values after Scaler:")
    logger.info(augmented_data_scaled[numerical_features])
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_features.joblib'))
    logger.info(f"Feature scaler saved to '{models_dir}/scaler_features.joblib'.")

    # Save the scaler for augmented data
    scaler_aug_path = os.path.abspath('../models/scaler_augmented.joblib')
    try:
        joblib.dump(scaler, scaler_aug_path)
        logger.info(f"Augmented scaler saved to '{scaler_aug_path}'.")
    except Exception as e:
        logger.error(f"Failed to save augmented scaler: {e}")
        raise

    return augmented_data_scaled, augmented_data_unscaled


def generate_fake_data(real_data_unscaled, percentage=0.5):
    """
    Generate fake stock data based on real data distribution.

    Parameters:
    - real_data_unscaled (DataFrame): The real stock data before scaling.
    - percentage (float): Percentage of fake data to generate relative to real data.

    Returns:
    - fake_df_unscaled (DataFrame): The generated fake stock data before scaling.
    """
    num_fake = int(len(real_data_unscaled) * percentage)
    fake_data = []

    for i in range(num_fake):
        fake_entry = {
            'Date': fake.date_between(start_date='-2y', end_date='today'),
            'Open': np.random.uniform(low=real_data_unscaled['Open'].min(), high=real_data_unscaled['Open'].max()),
            'High': np.random.uniform(low=real_data_unscaled['High'].min(), high=real_data_unscaled['High'].max()),
            'Low': np.random.uniform(low=real_data_unscaled['Low'].min(), high=real_data_unscaled['Low'].max()),
            'Close': np.random.uniform(low=real_data_unscaled['Close'].min(), high=real_data_unscaled['Close'].max()),
            'Volume': np.random.randint(low=int(real_data_unscaled['Volume'].min()), high=int(real_data_unscaled['Volume'].max())),
            'Dividends': np.random.uniform(low=real_data_unscaled['Dividends'].min(), high=real_data_unscaled['Dividends'].max()),
            'Stock Splits': np.random.uniform(low=real_data_unscaled['Stock Splits'].min(), high=real_data_unscaled['Stock Splits'].max()),
            'Ticker': np.random.choice(real_data_unscaled['Ticker'].unique()),
            'Avg_Sentiment_Score': np.random.uniform(-1.0, 1.0)
        }
        fake_data.append(fake_entry)

    fake_df_unscaled = pd.DataFrame(fake_data)
    fake_df_unscaled['Date'] = pd.to_datetime(fake_df_unscaled['Date'])  # Ensure 'Date' is datetime
    logger.info(f"Generated {num_fake} fake stock data entries.")
    return fake_df_unscaled


def perform_stock_analysis(stock_data):
    """
    Perform comprehensive statistical analysis on stock data.

    Parameters:
    - stock_data (DataFrame): The stock data to analyze

    Returns:
    - dict: Dictionary containing all analysis results
    """
    # Create copy to avoid modifying original data
    df = stock_data.copy()

    # Ensure numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

    # Initialize results dictionary
    analysis_results = {}

    # Basic statistics for each numeric column
    for column in numeric_columns:
        if column in df.columns:
            analysis_results[column] = {
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'variance': df[column].var(),
                'skewness': df[column].skew(),
                'kurtosis': df[column].kurtosis()
            }

    # Calculate correlations
    correlation_matrix = df[numeric_columns].corr()
    analysis_results['correlation_matrix'] = correlation_matrix

    return analysis_results


def display_paginated_dataframe_with_stats(df, rows_per_page=100):
    """
    Display a paginated dataframe with statistical analysis for each page.

    Parameters:
    - df (DataFrame): The dataframe to display
    - rows_per_page (int): Number of rows per page
    """
    # Calculate number of pages
    n_pages = len(df) // rows_per_page + (1 if len(df) % rows_per_page != 0 else 0)

    # Add a page number selectbox
    page_number = st.selectbox('Page', range(1, n_pages + 1))

    # Get the records for the current page
    start_idx = (page_number - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, len(df))
    page_data = df.iloc[start_idx:end_idx].copy()

    # Display the current page data
    st.dataframe(page_data)

    # Show page statistics
    st.write(f"Showing rows {start_idx + 1} to {end_idx} of {len(df)}")

    # Calculate and display statistics for the current page
    numeric_columns = page_data.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) > 0:
        st.write("### Statistics for Current Page")

        # Create tabs for different statistical views
        tab1, tab2 = st.tabs(["Basic Statistics", "Detailed Statistics"])

        with tab1:
            # Basic statistics in a more compact format
            basic_stats = pd.DataFrame({
                'Mean': page_data[numeric_columns].mean(),
                'Median': page_data[numeric_columns].median(),
                'Std Dev': page_data[numeric_columns].std(),
                'Min': page_data[numeric_columns].min(),
                'Max': page_data[numeric_columns].max()
            }).round(2)
            st.dataframe(basic_stats)

        with tab2:
            # More detailed statistics
            detailed_stats = pd.DataFrame({
                'Count': page_data[numeric_columns].count(),
                'Mean': page_data[numeric_columns].mean(),
                'Median': page_data[numeric_columns].median(),
                'Std Dev': page_data[numeric_columns].std(),
                'Variance': page_data[numeric_columns].var(),
                'Skewness': page_data[numeric_columns].skew(),
                'Kurtosis': page_data[numeric_columns].kurtosis(),
                'Min': page_data[numeric_columns].min(),
                '25%': page_data[numeric_columns].quantile(0.25),
                '75%': page_data[numeric_columns].quantile(0.75),
                'Max': page_data[numeric_columns].max()
            }).round(2)
            st.dataframe(detailed_stats)

            # Add download button for detailed statistics
            csv = detailed_stats.to_csv()
            st.download_button(
                label="Download Page Statistics CSV",
                data=csv,
                file_name=f"page_{page_number}_statistics.csv",
                mime="text/csv"
            )


def create_correlation_heatmap(correlation_matrix):
    """
    Create a correlation heatmap using plotly
    """
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    fig.update_layout(
        title='Correlation Heatmap of Stock Features',
        width=800,
        height=800
    )
    return fig


def create_statistical_plots(stock_data, column):
    """
    Create statistical plots for a given column
    """
    # Distribution plot
    fig_dist = px.histogram(
        stock_data,
        x=column,
        title=f'Distribution of {column}',
        nbins=50
    )
    fig_dist.add_trace(
        px.violin(stock_data, y=column).data[0]
    )

    # Box plot by ticker
    fig_box = px.box(
        stock_data,
        x='Ticker',
        y=column,
        title=f'Box Plot of {column} by Ticker'
    )

    return fig_dist, fig_box

def calculate_sentiment_score(sentiment):
    """
    Calculate sentiment score based on sentiment label.

    Parameters:
    - sentiment (str): Sentiment label (e.g., 'positive', 'neutral', 'negative').

    Returns:
    - score (int): Numerical sentiment score.
    """
    sentiment_mapping = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }
    score = sentiment_mapping.get(sentiment.lower(), 0)
    return score


def train_models(cleaned_stock_data_scaled, sentiment_data):
    """
    Train machine learning models to predict stock 'Close' price based on features and sentiment.

    Parameters:
    - cleaned_stock_data_scaled (DataFrame): The cleaned and scaled real stock data.
    - sentiment_data (DataFrame): The sentiment data with scores.

    Returns:
    - None (models are saved to disk)
    """
    try:
        # Merge stock data with sentiment data on 'Ticker'
        sentiment_agg = sentiment_data.groupby('Ticker')['score'].mean().reset_index()
        sentiment_agg.rename(columns={'score': 'Avg_Sentiment_Score'}, inplace=True)

        # Merge with stock data
        df = pd.DataFrame(cleaned_stock_data_scaled)
        #df = pd.merge(cleaned_stock_data_scaled, sentiment_agg, on='Ticker', how='left')
        #df['Avg_Sentiment_Score'].fillna(0, inplace=True)  # Handle missing sentiment scores

        # Define features and target
        features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Avg_Sentiment_Score']  # Removed 'Close'
        target_regression = 'Close'  # Predicting 'Close' price
        target_classification = 'Close_Class'  # Binary classification: Increase or Decrease

        # Create classification target
        df[target_classification] = df['Close'].diff().fillna(0)
        df[target_classification] = df[target_classification].apply(lambda x: 1 if x > 0 else 0)

        # Drop rows with NaN in target
        df.dropna(subset=[target_regression, target_classification], inplace=True)

        # Split data into features and targets
        X_reg = df[features]
        y_reg = df[target_regression]
        X_clf = df[features]
        y_clf = df[target_classification]

        # Split into training and testing sets
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2,
                                                                            random_state=42)
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2,
                                                                            random_state=42)

        # Initialize models
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        svm_reg = SVR(kernel='rbf')

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_clf = SVC(kernel='rbf', probability=True)

        # Train regression models
        rf_reg.fit(X_train_reg, y_train_reg)
        svm_reg.fit(X_train_reg, y_train_reg)
        logger.info("Trained Random Forest Regressor and Support Vector Regressor.")

        # Train classification models
        rf_clf.fit(X_train_clf, y_train_clf)
        svm_clf.fit(X_train_clf, y_train_clf)
        logger.info("Trained Random Forest Classifier and Support Vector Classifier.")

        # Evaluate regression models
        rf_reg_pred = rf_reg.predict(X_test_reg)
        svm_reg_pred = svm_reg.predict(X_test_reg)

        rf_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_reg_pred))
        svm_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, svm_reg_pred))

        logger.info(f"Random Forest Regressor RMSE: {rf_reg_rmse}")
        logger.info(f"SVM Regressor RMSE: {svm_reg_rmse}")

        # Evaluate classification models
        rf_clf_pred = rf_clf.predict(X_test_clf)
        svm_clf_pred = svm_clf.predict(X_test_clf)

        rf_clf_acc = accuracy_score(y_test_clf, rf_clf_pred)
        svm_clf_acc = accuracy_score(y_test_clf, svm_clf_pred)

        logger.info(f"Random Forest Classifier Accuracy: {rf_clf_acc}")
        logger.info(f"SVM Classifier Accuracy: {svm_clf_acc}")

        # Optionally, log classification report
        logger.info("Random Forest Classifier Report:")
        logger.info(classification_report(y_test_clf, rf_clf_pred))

        logger.info("SVM Classifier Report:")
        logger.info(classification_report(y_test_clf, svm_clf_pred))

        # Create a dictionary to store evaluation metrics
        metrics = {
            'Regression': {
                'Random Forest Regressor RMSE': rf_reg_rmse,
                'SVM Regressor RMSE': svm_reg_rmse
            },
            'Classification': {
                'Random Forest Classifier Accuracy': rf_clf_acc,
                'SVM Classifier Accuracy': svm_clf_acc
            }
        }

        # Save metrics to a CSV file
        metrics_df = pd.DataFrame(metrics)
        metrics_dir = os.path.abspath('../models/metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_dir, 'model_metrics.csv'), index=False)
        logger.info(f"Model evaluation metrics saved to '{metrics_dir}/model_metrics.csv'.")

        # Save models to disk
        models_dir = os.path.abspath('../models')
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(rf_reg, os.path.join(models_dir, 'random_forest_regressor.joblib'))
        joblib.dump(svm_reg, os.path.join(models_dir, 'svm_regressor.joblib'))
        joblib.dump(rf_clf, os.path.join(models_dir, 'random_forest_classifier.joblib'))
        joblib.dump(svm_clf, os.path.join(models_dir, 'svm_classifier.joblib'))
        logger.info(f"Trained models saved to '{models_dir}'.")


    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise