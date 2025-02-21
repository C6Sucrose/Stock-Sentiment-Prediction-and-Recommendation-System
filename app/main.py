# app/main.py

import streamlit as st
from twisted.python.util import println

from utils.data_processing import load_and_prepare_data, display_paginated_dataframe_with_stats, perform_stock_analysis, \
    create_correlation_heatmap, create_statistical_plots
import pandas as pd
import plotly.express as px
from math import ceil
import joblib
import os
from sklearn.preprocessing import StandardScaler
import logging
from streamlit_chat import message
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="Stock Prediction And Sentiment Recommendation System", layout="wide")

# Helper function for paginated dataframes
def display_paginated_dataframe(df, rows_per_page=100):
    """
    Display a pandas dataframe with pagination controls.

    Parameters:
    - df (DataFrame): The dataframe to display.
    - rows_per_page (int): Number of rows to display per page.
    """
    if df.empty:
        st.warning("No data available to display.")
        return
    total_pages = ceil(len(df) / rows_per_page) if len(df) else 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    st.dataframe(df.iloc[start_idx:end_idx])

# Caching data loading to optimize performance
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_data(fetch, manual_refresh):
    return load_and_prepare_data(fetch=fetch, manual_refresh=manual_refresh)

# Sidebar for navigation
st.sidebar.title("Stock Sentiment Recommender")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Visualization", "Model Predictions", "Chatbot"])

# Sidebar for Data Management
st.sidebar.header("Data Management")
manual_refresh = st.sidebar.button("Manual Data Refresh")
fetch_data = True  # Set to True to enable automatic checks

# Display spinner while loading data
with st.spinner('Loading data...'):
    try:
        cleaned_stock_data_scaled, augmented_data_scaled, sentiment_data, cleaned_stock_data_unscaled, augmented_data_unscaled = get_data(fetch=fetch_data, manual_refresh=manual_refresh)
        st.success('Data loaded successfully!')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()  # Stop execution if data fails to load

# Load trained models
def load_models():
    """
    Load trained machine learning models from the models directory.

    Returns:
    - models (dict): Dictionary containing loaded models.
    """
    models = {}
    models_dir = os.path.abspath('../models')
    model_files = {
        'rf_reg': 'random_forest_regressor.joblib',
        'svm_reg': 'svm_regressor.joblib',
        'rf_clf': 'random_forest_classifier.joblib',
        'svm_clf': 'svm_classifier.joblib'
    }

    for key, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[key] = joblib.load(model_path)
            st.sidebar.success(f"Loaded {filename}")
        else:
            st.sidebar.error(f"Model file '{filename}' not found in '{models_dir}'. Please train the models first.")
    return models

models = load_models()

# Function to load evaluation metrics
def load_metrics():
    """
    Load model evaluation metrics from the metrics directory.

    Returns:
    - metrics_df (DataFrame): DataFrame containing evaluation metrics.
    """
    metrics_path = os.path.abspath('../models/metrics/model_metrics.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        return metrics_df
    else:
        logger.warning("Model metrics file not found.")
        return pd.DataFrame()

# Function to load Close scaler
def load_close_scaler():
    """
    Load the scaler used for the 'Close' price.

    Returns:
    - scaler_close (StandardScaler): The loaded scaler object for 'Close'.
    """
    scaler_close_path = os.path.abspath('../models/scaler_close.joblib')
    if os.path.exists(scaler_close_path):
        scaler_close = joblib.load(scaler_close_path)
        logger.info("scaler_close Function loaded scale: " + str(scaler_close))
        return scaler_close
    else:
        st.warning("Close scaler file not found.")
        return None

scaler_close = load_close_scaler()

def load_features_scaler():
    """
    Load the scaler used for scaling the input features.

    Returns:
    - scaler_features (StandardScaler): The loaded scaler object for features.
    """
    scaler_features_path = os.path.abspath('../models/scaler_features.joblib')
    if os.path.exists(scaler_features_path):
        scaler_features = joblib.load(scaler_features_path)
        return scaler_features
    else:
        st.warning("Feature scaler not found.")
        return None


scaler_features = load_features_scaler()


# Function to send message to Rasa and get response
def get_rasa_response(message_text):
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {"sender": "streamlit_user", "message": message_text}
    try:
        response = requests.post(rasa_url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return [{"text": "⚠️ Error: Could not process your request."}]
    except Exception as e:
        return [{"text": f"⚠️ Error: Could not connect to the chatbot service. ({str(e)})"}]






# Initialize session state for chatbot history
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []



# Page Routing
if page == "Home":
    st.title("Welcome to the Stock Sentiment Recommender")
    st.write("""
    The **Stock Sentiment Recommender** is a comprehensive tool designed to analyze stock market trends based on historical data and sentiment analysis from social media platforms like Reddit. Navigate through the app using the sidebar to explore data visualizations, view predictive models, or interact with our intelligent chatbot.
    """)
    st.image("https://via.placeholder.com/800x400.png?text=Stock+Sentiment+Recommender", use_column_width=True)

elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("Explore interactive visualizations of stock and sentiment data to gain insightful trends and patterns.")

    # Toggle for Stock Data
    if st.checkbox("Show Stock Data"):
        st.subheader("Stock Data (Augmented)")
        display_paginated_dataframe_with_stats(augmented_data_scaled, rows_per_page=100)

    # Toggle for Sentiment Data
    if st.checkbox("Show Sentiment Data"):
        st.subheader("Sentiment Data")
        display_paginated_dataframe_with_stats(sentiment_data, rows_per_page=100)

    st.markdown("---")  # Separator

    # Visualization: Closing Prices Over Time (Augmented Data)
    st.subheader("Closing Prices Over Time (Augmented Data)")
    if not augmented_data_unscaled.empty:
        try:
            # Ensure data is sorted by Date
            augmented_data_unscaled_sorted = augmented_data_unscaled.sort_values(by='Date')
            fig_augmented = px.line()
            tickers = augmented_data_unscaled_sorted['Ticker'].unique()
            for ticker in tickers:
                ticker_data = augmented_data_unscaled_sorted[augmented_data_unscaled_sorted['Ticker'] == ticker]
                fig_augmented.add_scatter(x=ticker_data['Date'], y=ticker_data['Close'], mode='lines', name=ticker)
            fig_augmented.update_layout(
                title='Stock Closing Prices Over Time (Augmented Data)',
                xaxis_title='Date',
                yaxis_title='Closing Price',
                legend_title='Ticker',
                hovermode="x unified"
            )
            st.plotly_chart(fig_augmented, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating Closing Prices Over Time (Augmented Data) chart: {e}")
    else:
        st.warning("Augmented stock data is empty. Cannot generate Closing Prices chart.")

    # Visualization: Closing Prices Over Time (Real Data)
    st.subheader("Closing Prices Over Time (Real Data)")
    if not cleaned_stock_data_unscaled.empty:
        try:
            # Ensure data is sorted by Date
            real_data_sorted = cleaned_stock_data_unscaled.sort_values(by='Date')
            fig_real = px.line()
            tickers_real = real_data_sorted['Ticker'].unique()
            for ticker in tickers_real:
                ticker_data = real_data_sorted[real_data_sorted['Ticker'] == ticker]
                fig_real.add_scatter(x=ticker_data['Date'], y=ticker_data['Close'], mode='lines', name=ticker)
            fig_real.update_layout(
                title='Stock Closing Prices Over Time (Real Data)',
                xaxis_title='Date',
                yaxis_title='Closing Price',
                legend_title='Ticker',
                hovermode="x unified"
            )
            st.plotly_chart(fig_real, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating Closing Prices Over Time (Real Data) chart: {e}")
    else:
        st.warning("Real stock data is empty. Cannot generate Closing Prices chart.")

    # Visualization: Sentiment Score Distribution (All Sentiment Data)
    st.subheader("Sentiment Score Distribution (All Sentiment Data)")
    if 'score' in sentiment_data.columns and sentiment_data['score'].notnull().any():
        try:
            fig_sentiment_all = px.histogram(
                sentiment_data,
                x='score',
                nbins=3,  # Since scores are -1, 0, 1
                title='Distribution of Sentiment Scores (All Sentiment Data)',
                labels={'score': 'Sentiment Score'},
                color_discrete_sequence=['#636EFA']
            )
            fig_sentiment_all.update_layout(
                xaxis_title='Sentiment Score',
                yaxis_title='Count',
                bargap=0.2
            )
            st.plotly_chart(fig_sentiment_all, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating Sentiment Score Distribution chart: {e}")
    else:
        st.warning("The sentiment data does not contain a valid 'score' column or all scores are NaN.")

    st.markdown("---")  # Separator

    # Statistical Analysis Section
    st.header("Overall Statistical Analysis")

    # Add radio button to choose between real and augmented data
    data_choice = st.radio(
        "Choose data for overall statistical analysis:",
        ("Real Data", "Augmented Data")
    )

    if data_choice == "Real Data":
        if not cleaned_stock_data_unscaled.empty:
            # Display correlation analysis
            st.subheader("Correlation Analysis")
            analysis_results = perform_stock_analysis(cleaned_stock_data_unscaled)
            fig_corr = create_correlation_heatmap(analysis_results['correlation_matrix'])
            st.plotly_chart(fig_corr, use_container_width=True)

            # Distribution analysis
            st.subheader("Distribution Analysis")
            selected_column = st.selectbox(
                "Select Feature to Analyze",
                ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            )
            fig_dist, fig_box = create_statistical_plots(cleaned_stock_data_unscaled, selected_column)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Real stock data is empty. Cannot perform statistical analysis.")
    else:  # Augmented Data
        if not augmented_data_unscaled.empty:
            # Display correlation analysis
            st.subheader("Correlation Analysis")
            analysis_results = perform_stock_analysis(augmented_data_unscaled)
            fig_corr = create_correlation_heatmap(analysis_results['correlation_matrix'])
            st.plotly_chart(fig_corr, use_container_width=True)

            # Distribution analysis
            st.subheader("Distribution Analysis")
            selected_column = st.selectbox(
                "Select Feature to Analyze",
                ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            )
            fig_dist, fig_box = create_statistical_plots(augmented_data_unscaled, selected_column)
            st.plotly_chart(fig_dist, use_container_width=True)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Augmented stock data is empty. Cannot perform statistical analysis.")





elif page == "Model Predictions":
    st.title("Model Predictions")
    st.write("Predict stock movements based on sentiment and historical data.")

    # Display Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    metrics_df = load_metrics()
    if not metrics_df.empty:
        st.table(metrics_df)
    else:
        st.warning("No evaluation metrics available. Please train the models first.")

    if not models:
        st.error("No models loaded. Please ensure models are trained and placed in the '../models' directory.")
    else:
        # Select model type
        model_type = st.selectbox("Select Model Type", ["Regression", "Classification"])

        if model_type == "Regression":
            # Select which regression model to use
            reg_model_choice = st.selectbox("Choose Regression Model", ["Random Forest Regressor", "Support Vector Regressor"])

            if reg_model_choice == "Random Forest Regressor":
                selected_model = models.get('rf_reg')
            else:
                selected_model = models.get('svm_reg')

            # Input widgets for features
            st.sidebar.header("Input Features")

            # Allow user to select Ticker
            available_tickers = cleaned_stock_data_scaled['Ticker'].unique()
            selected_ticker = st.sidebar.selectbox("Select Fine-Tuning Ticker", available_tickers)

            # Filter the unscaled data for the selected ticker to get realistic ranges
            ticker_unscaled = cleaned_stock_data_unscaled[cleaned_stock_data_unscaled['Ticker'] == selected_ticker]
            if ticker_unscaled.empty:
                st.warning("Selected ticker has no unscaled data available.")
                st.stop()

            # Calculate Average Sentiment Score for the selected ticker
            avg_sentiment_score = sentiment_data[sentiment_data['Ticker'] == selected_ticker]['score'].mean()
            if pd.isna(avg_sentiment_score):
                avg_sentiment_score = 0.0  # Default value if no sentiment data available
                st.sidebar.warning("No sentiment data available for the selected ticker. Using default sentiment score of 0.0.")

            # Define ranges based on selected ticker's unscaled data
            open_min, open_max = ticker_unscaled['Open'].min(), ticker_unscaled['Open'].max()
            high_min, high_max = ticker_unscaled['High'].min(), ticker_unscaled['High'].max()
            low_min, low_max = ticker_unscaled['Low'].min(), ticker_unscaled['Low'].max()
            # Convert volume to thousands
            volume_min = float(ticker_unscaled['Volume'].min()) / 1_000  # Convert to thousands
            volume_max = float(ticker_unscaled['Volume'].max()) / 1_000
            volume_mean = float(ticker_unscaled['Volume'].mean()) / 1_000
            dividends_min, dividends_max = ticker_unscaled['Dividends'].min(), ticker_unscaled['Dividends'].max()
            stock_splits_min, stock_splits_max = ticker_unscaled['Stock Splits'].min(), ticker_unscaled['Stock Splits'].max()



            open_input = st.sidebar.slider("Open Price", float(open_min), float(open_max), float(ticker_unscaled['Open'].mean()))
            high_input = st.sidebar.slider("High Price", float(high_min), float(high_max), float(ticker_unscaled['High'].mean()))
            low_input = st.sidebar.slider("Low Price", float(low_min), float(low_max), float(ticker_unscaled['Low'].mean()))
            # Display volume in thousands (K)
            volume_input = st.sidebar.slider(
                "Volume (in K)",
                volume_min,
                volume_max,
                volume_mean
            ) * 1_000  # Convert back to original scale for calculations
            dividends_input = st.sidebar.slider("Dividends", float(dividends_min), float(dividends_max), float(ticker_unscaled['Dividends'].mean()))
            stock_splits_input = st.sidebar.slider("Stock Splits", float(stock_splits_min), float(stock_splits_max), float(ticker_unscaled['Stock Splits'].mean()))

            # Prepare input data for prediction (Exclude 'Close')
            input_data = pd.DataFrame({
                'Open': [open_input],
                'High': [high_input],
                'Low': [low_input],
                'Volume': [volume_input],
                'Dividends': [dividends_input],
                'Stock Splits': [stock_splits_input],
                'Avg_Sentiment_Score': [avg_sentiment_score]
            })

            # Make prediction
            if st.sidebar.button("Predict"):
                if scaler_close and scaler_features:
                    try:
                        # Load scaler for features (ensure it's fitted on training data)
                        scaler_features_path = os.path.abspath('../models/scaler_features.joblib')  # Save this scaler during training
                        if os.path.exists(scaler_features_path):
                            scaler_features = joblib.load(scaler_features_path)
                        else:
                            st.error("Feature scaler not found. Please ensure it's saved correctly during training.")
                            st.stop()

                        # Scale the input data
                        input_data_scaled = scaler_features.transform(input_data)

                        # Predict using the regression model
                        prediction_scaled = selected_model.predict(input_data_scaled)[0]

                        # Inverse transform the prediction to original scale
                        #prediction_original = scaler_close.inverse_transform([[prediction_scaled]])[0][0]

                        st.subheader(f"Predicted Closing Price for {selected_ticker} (Original Scale): ${prediction_scaled:.2f}")

                        # Visualization: Predicted vs Current Price
                        current_price = ticker_unscaled[ticker_unscaled['Date'] == ticker_unscaled['Date'].max()]['Close'].values[0]
                        price_diff = prediction_scaled - current_price
                        logger.info("current price:" + str(current_price))
                        logger.info("predicted price:" + str(prediction_scaled))


                        if price_diff > 0:
                            st.success(f"The model predicts an **increase** of ${price_diff:.2f} in the closing price compared to the current market price of ${current_price:.2f}. Recommendation: **BUY**")
                            st.markdown("## :chart_with_upwards_trend: Price Prediction")
                            st.image("https://i.imgur.com/UpTrend.png", width=50)  # Example up arrow
                        elif price_diff < 0:
                            st.error(f"The model predicts a **decrease** of ${abs(price_diff):.2f} in the closing price compared to the current market price of ${current_price:.2f}.  Recommendation: **HOLD**")
                            st.markdown("## :chart_with_downwards_trend: Price Prediction")
                            st.image("https://i.imgur.com/DnTrend.png", width=50)  # Example down arrow
                        elif -50 < price_diff < 0:
                            st.error(f"The model predicts a **decrease** of ${abs(price_diff):.2f} in the closing price compared to the current market price of ${current_price:.2f}.  Recommendation: **SELL**")
                            st.markdown("## :chart_with_downwards_trend: Price Prediction")
                            st.image("https://i.imgur.com/DnTrend.png", width=50)  # Example down arrow
                        else:
                            st.info("The model predicts no change in the closing price.  Recommendation: **HOLD**")
                            st.markdown("## :straight_ruler: Price Prediction")
                            st.image("https://i.imgur.com/NoChange.png", width=50)  # Example no change

                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                else:
                    st.error("Close scaler not available. Cannot inverse transform the prediction.")

        elif model_type == "Classification":
            # Existing Classification Code...
            clf_model_choice = st.selectbox("Choose Classification Model", ["Random Forest Classifier", "Support Vector Classifier"])

            if clf_model_choice == "Random Forest Classifier":
                selected_clf_model = models.get('rf_clf')
            else:
                selected_clf_model = models.get('svm_clf')

            st.sidebar.header("Input Features")

            # Allow user to select Ticker
            available_tickers = cleaned_stock_data_scaled['Ticker'].unique()
            selected_ticker = st.sidebar.selectbox("Select Ticker", available_tickers)

            # Filter the unscaled data for the selected ticker to get realistic ranges
            ticker_unscaled = cleaned_stock_data_unscaled[cleaned_stock_data_unscaled['Ticker'] == selected_ticker]
            if ticker_unscaled.empty:
                st.warning("Selected ticker has no unscaled data available.")
                st.stop()

            # Calculate Average Sentiment Score for the selected ticker
            avg_sentiment_score = sentiment_data[sentiment_data['Ticker'] == selected_ticker]['score'].mean()
            if pd.isna(avg_sentiment_score):
                avg_sentiment_score = 0.0  # Default value if no sentiment data available
                st.sidebar.warning("No sentiment data available for the selected ticker. Using default sentiment score of 0.0.")

            # Define ranges based on selected ticker's unscaled data
            open_min, open_max = ticker_unscaled['Open'].min(), ticker_unscaled['Open'].max()
            high_min, high_max = ticker_unscaled['High'].min(), ticker_unscaled['High'].max()
            low_min, low_max = ticker_unscaled['Low'].min(), ticker_unscaled['Low'].max()
            # Convert volume to thousands
            volume_min = float(ticker_unscaled['Volume'].min()) / 1_000  # Convert to thousands
            volume_max = float(ticker_unscaled['Volume'].max()) / 1_000
            volume_mean = float(ticker_unscaled['Volume'].mean()) / 1_000
            dividends_min, dividends_max = ticker_unscaled['Dividends'].min(), ticker_unscaled['Dividends'].max()
            stock_splits_min, stock_splits_max = ticker_unscaled['Stock Splits'].min(), ticker_unscaled['Stock Splits'].max()

            open_input = st.sidebar.slider("Open Price", float(open_min), float(open_max), float(ticker_unscaled['Open'].mean()))
            high_input = st.sidebar.slider("High Price", float(high_min), float(high_max), float(ticker_unscaled['High'].mean()))
            low_input = st.sidebar.slider("Low Price", float(low_min), float(low_max), float(ticker_unscaled['Low'].mean()))
            # Display volume in thousands (K)
            volume_input = st.sidebar.slider(
                "Volume (in K)",
                volume_min,
                volume_max,
                volume_mean
            ) * 1_000  # Convert back to original scale for calculations
            dividends_input = st.sidebar.slider("Dividends", float(dividends_min), float(dividends_max), float(ticker_unscaled['Dividends'].mean()))
            stock_splits_input = st.sidebar.slider("Stock Splits", float(stock_splits_min), float(stock_splits_max), float(ticker_unscaled['Stock Splits'].mean()))

            # Prepare input data for prediction (Exclude 'Close')
            input_data = pd.DataFrame({
                'Open': [open_input],
                'High': [high_input],
                'Low': [low_input],
                'Volume': [volume_input],
                'Dividends': [dividends_input],
                'Stock Splits': [stock_splits_input],
                'Avg_Sentiment_Score': [avg_sentiment_score]
            })

            # Make prediction
            if st.sidebar.button("Predict"):
                try:
                    # Load scaler for features (ensure it's fitted on training data)
                    scaler_features_path = os.path.abspath('../models/scaler_features.joblib')  # Save this scaler during training
                    if os.path.exists(scaler_features_path):
                        scaler_features = joblib.load(scaler_features_path)
                    else:
                        st.error("Feature scaler not found. Please ensure it's saved correctly during training.")
                        st.stop()

                    # Scale the input data
                    input_data_scaled = scaler_features.transform(input_data)

                    # Predict using the classification model
                    prediction = selected_clf_model.predict(input_data_scaled)[0]
                    prediction_proba = selected_clf_model.predict_proba(input_data_scaled)[0]

                    sentiment = 'Increase' if prediction == 1 else 'Decrease'
                    proba = max(prediction_proba) * 100

                    st.subheader(f"Predicted Stock Movement for {selected_ticker}: {sentiment} ({proba:.2f}% Confidence)")

                    # Visualization: Prediction Confidence
                    fig_proba = px.bar(
                        x=['Decrease', 'Increase'],
                        y=prediction_proba,
                        labels={'x': 'Movement', 'y': 'Probability'},
                        title='Prediction Confidence',
                        color=['#EF553B', '#636EFA']
                    )
                    st.plotly_chart(fig_proba, use_container_width=True)

                except Exception as e:
                    st.error(f"Error making prediction: {e}")

elif page == "Chatbot":
    st.title("📈 Stock Analysis Chatbot")

    # Create two columns for the main layout
    main_col1, main_col2 = st.columns([2, 1])

    with main_col1:
        st.write("Get real-time insights about stocks through our AI-powered chatbot.")

        # Add Clear Chat button and chat container in a row
        clear_col, _ = st.columns([1, 4])
        with clear_col:
            if st.button("🗑️ Clear Chat"):
                st.session_state.generated = []
                st.session_state.past = []
                st.experimental_rerun()

        # Chat interface container
        chat_container = st.container()

        # Initialize the input state if not exists
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        # Input area with send button
        with st.container():
            input_col1, input_col2 = st.columns([5, 1])
            with input_col1:
                user_input = st.text_input(
                    "Ask about stocks:",
                    key="user_input",
                    placeholder="Try asking about stock predictions, sentiment, or recommendations..."
                )
            with input_col2:
                send_button = st.button("Send", use_container_width=True)

    # Process user input when send button is clicked
    if send_button and user_input:
        st.session_state.past.append(user_input)
        response = get_rasa_response(user_input)
        if response:
            st.session_state.generated.append(response[0].get("text", "I'm sorry, I couldn't understand that."))
        else:
            st.session_state.generated.append("No response received from the chatbot.")

        # Instead of directly modifying session state, use a callback
        st.experimental_rerun()

    # Sidebar with quick actions and stock selector
    with main_col2:
        st.subheader("Quick Actions")

        # Stock selector
        selected_stock = st.selectbox(
            "Select Stock",
            ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"],
            key="stock_selector"
        )

        # Quick action buttons with custom styling
        st.markdown("""
        <style>
        div.stButton > button {
            width: 100%;
            margin: 5px 0;
            background-color: #000000;
            border: 1px solid #000001;
        }
        div.stButton > button:hover {
            background-color: #e0e3e9;
            border: 1px solid #d0d3d9;
        }
        .chat-message-bot {
            background-color: #000000;
            padding: 15px;
            border-radius: 15px;
            margin: 5px 0;
            display: flex;
            flex-direction: column;
        }
        .chat-message-user {
            background-color: #2196F3;
            color: white;
            padding: 15px;
            border-radius: 15px;
            margin: 5px 0;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        .metadata {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Quick action buttons in a grid
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Details", help=f"Get details about {selected_stock}"):
                if 'past' not in st.session_state:
                    st.session_state['past'] = []
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = []
                query = f"Tell me about {selected_stock}"
                st.session_state.past.append(query)
                response = get_rasa_response(query)
                if response:
                    st.session_state.generated.append(response[0].get("text", "Error processing request"))

            if st.button("📈 Predict", help=f"Get price prediction for {selected_stock}"):
                if 'past' not in st.session_state:
                    st.session_state['past'] = []
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = []
                query = f"Predict {selected_stock}"
                st.session_state.past.append(query)
                response = get_rasa_response(query)
                if response:
                    st.session_state.generated.append(response[0].get("text", "Error processing request"))

        with col2:
            if st.button("🎯 Sentiment", help=f"Get sentiment analysis for {selected_stock}"):
                if 'past' not in st.session_state:
                    st.session_state['past'] = []
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = []
                query = f"What's the sentiment for {selected_stock}"
                st.session_state.past.append(query)
                response = get_rasa_response(query)
                if response:
                    st.session_state.generated.append(response[0].get("text", "Error processing request"))

            if st.button("💡 Recommend", help="Get stock recommendations"):
                if 'past' not in st.session_state:
                    st.session_state['past'] = []
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = []
                query = "Recommend stocks"
                st.session_state.past.append(query)
                response = get_rasa_response(query)
                if response:
                    st.session_state.generated.append(response[0].get("text", "Error processing request"))

        # Add information about supported stocks
        with st.expander("ℹ️ Supported Stocks"):
            st.write("""
            Currently supporting:
            - AAPL (Apple Inc.)
            - GOOGL (Alphabet Inc.)
            - MSFT (Microsoft Corporation)
            - NVDA (NVIDIA Corporation)
            - TSLA (Tesla, Inc.)
            """)

    # Initialize session state for chat history if not exists
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []


    # Display chat history
    with chat_container:
        for i in range(len(st.session_state['generated'])):
            # User message
            st.markdown(f'<div class="chat-message-user">{st.session_state["past"][i]}</div>',
                        unsafe_allow_html=True)

            # Bot message with potential metadata extraction
            bot_response = st.session_state['generated'][i]

            # Check if the response contains structured data
            if "**" in bot_response:  # Indicates formatted data
                # Create metrics container
                metrics_cols = st.columns(4)

                # Extract and display metrics if present
                try:
                    if "Close:" in bot_response:
                        price = bot_response.split("Close:")[1].split("\n")[0].strip()
                        metrics_cols[0].metric("Current Price", price)
                    if "Volume:" in bot_response:
                        volume = bot_response.split("Volume:")[1].split("\n")[0].strip()
                        metrics_cols[1].metric("Volume", volume)
                    if "predicted price:" in bot_response.lower():
                        pred_price = bot_response.lower().split("predicted price:")[1].split("\n")[0].strip()
                        metrics_cols[2].metric("Predicted Price", pred_price)
                    if "confidence:" in bot_response.lower():
                        conf = bot_response.lower().split("confidence:")[1].split("\n")[0].strip()
                        metrics_cols[3].metric("Confidence", conf)
                except Exception as e:
                    st.error(f"Error parsing metrics: {str(e)}")

            # Display the full response
            st.markdown(f'<div class="chat-message-bot">{bot_response}</div>',
                        unsafe_allow_html=True)

            # Add spacing between messages
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

    # Add a helpful note at the bottom
    st.markdown("---")
    st.caption("""
    💡 **Tips:**
    - Use the Quick Actions panel for common queries
    - Type custom questions about any supported stock
    - Ask about predictions, sentiment analysis, or get recommendations
    """)

# Footer
st.markdown("""
---
This Stock Prediction and Recommendendation System does not provide Financial Advice. Please do not consider it as such. 
""")