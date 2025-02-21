Ali, Huraira, 22304705

# Stock Sentiment Recommender

https://mygit.th-deg.de/ha06705/stock-sentiment-prediction-recommendation-system

https://mygit.th-deg.de/ha06705/stock-sentiment-prediction-recommendation-system/-/wikis/home


- [Project Description](#project-description)
- [Installation](#installation)
- [Data](#data)
- [Basic Usage](#basic-usage)
- [Implementation of Requests](#implementation-of-requests)
- [Work Done](#work-done)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

# Project Description

The **Stock Sentiment Recommender** is a multi-page web application designed to analyze and predict stock trends based on sentiment analysis. Leveraging real-time data, machine learning models, and interactive visualizations, the system provides users with actionable insights to inform their investment decisions.

# Installation

### Prerequisites

- Python==3.8.0(Rasa Framework has a bug which causes it to fail version check during install on Python 3.9+)
- Git
- streamlit==1.22.0
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.2.2
- matplotlib==3.7.2
- seaborn==0.12.2
- rasa==3.5.14
- rasa-sdk==3.5.1
- plotly==5.15.0
- altair==4.2.2
- requests==2.31.0
- joblib==1.3.2
- flask==2.3.2
- streamlit-chat==0.1.1
- python-dotenv==0.21.0
- sqlalchemy<2.0
- streamlit-chat==0.1.1

### Steps

Either clone the repository or download and extract it in your local project folder.

1. **Clone the Repository:**
   ```
   bash
   git clone https://mygit.th-deg.de/ha06705/stock-sentiment-prediction-recommendation-system.git
   ```

2. **Navigate to Project Directory**

3. **Create and Activate Virtual Environment:**
```
python -m venv venv
```
* On macOS/Linux:
    ```
    source venv/Scripts/activate
    ```
* On Windows:
    ```
    venv\Scripts\activate
    ```

4. **Install Dependencies:**
```
pip install -r requirements.txt
```



## Features

- **Interactive Data Visualization:** Explore stock data through dynamic charts and plots.
- **Machine Learning Predictions:** Utilize advanced algorithms to forecast stock performance.
- **Sentiment Analysis:** Analyze market sentiment from various data sources.
- **Chatbot Integration:** Engage with a Rasa-powered chatbot for personalized recommendations.
- **Real-Time Data:** Access up-to-date stock information and trends.

# Data

All data, both for sentiment and for stocks, is fetched in real time using Reddit and Yahoo finance respectively. If the data is over 1 hour old, it is fetched again.

### Outlier Handling and Data Augmentation Strategy
### Outlier Detection and Treatment
The code uses the Interquartile Range (IQR) method for handling outliers in the stock data:

1. IQR Method Implementation

- Calculates Q1 (25th percentile) and Q3 (75th percentile)
- Determines IQR = Q3 - Q1
- Sets boundaries:

- - Lower bound = Q1 - 1.5 * IQR
- - Upper bound = Q3 + 1.5 * IQR


- Caps values outside these bounds to the nearest boundary


3. Columns Processed

- Open
- High
- Low
- Close
- Volume
- Dividends
- Stock Splits


3. Preservation Strategy

- Original data is preserved in an unscaled copy
- Outlier capping is performed before scaling
- Maintains data relationships while reducing extreme values



### Fake Data Generation

1. Volume Control

- Uses percentage-based generation (default 50% of original data size)
- Configurable through the percentage parameter


2. Data Distribution Matching

- Generates values within the bounds of real data:

- - Uses min/max ranges from real data for each feature
- - Maintains realistic value ranges for each stock metric




3. Feature Generation Strategy
```
pythonCopyfake_entry = {
    'Date': Faker-generated date (-5y to today)
    'Open': uniform(min, max) of real data
    'High': uniform(min, max) of real data
    'Low': uniform(min, max) of real data
    'Close': uniform(min, max) of real data
    'Volume': randint(min, max) of real data
    'Dividends': uniform(min, max) of real data
    'Stock Splits': uniform(min, max) of real data
    'Ticker': random choice from real tickers
    'Avg_Sentiment_Score': uniform(-1.0, 1.0)
}
```

4. Data Integrity Measures

- Datetime conversion for consistency
- Random but realistic sentiment scores
- Preserves ticker distribution from original data


5. Post-Generation Processing

- Concatenation with real data
- Scaling application to maintain consistency
- Separate scalers for 'Close' price and other features


# Basic Usage

### Running the Streamlit App
1. **Activate Virtual Environment:**
```
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
2. **Launch the App:**
We will first run the rasa actions server, then the rasa api server and then streamlit afterwards.
In the Project Directory:
In a bash terminal:
```
cd rasa_bot
rasa run actions
```
In a new bash terminal:
```
cd rasa_bot
rasa run --enable-api --cors "*" --debug
```
In a new bash terminal:
```
cd app
streamlit run main.py
```

* This command starts the Streamlit server and opens the application in your default web browser.

### Using the Chatbot
Interact with the integrated Rasa chatbot within the Streamlit app to receive personalized stock recommendations and sentiment insights.

On the right of the chatbot are the quick shortcuts as well as a dropdown to select one of the 5 available stocks right now.

# Implementation of the Requests
## Implementation of the Requests in Main

## 1. Data Loading and Management

### Caching and Performance Optimization
The application uses Streamlit's `@st.cache_data` decorator for the `get_data()` function, which:
- Caches loaded data for 1 hour
- Allows manual data refresh through the sidebar button
- Optimizes performance by preventing repeated data loading

### Data Loading Process
```python
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_data(fetch, manual_refresh):
    return load_and_prepare_data(fetch=fetch, manual_refresh=manual_refresh)
```

Key aspects:
- Supports automatic and manual data fetching
- Uses a time-to-live (TTL) of 3600 seconds (1 hour)
- Prevents unnecessary repeated data processing

## 2. Model Management

### Model Loading Strategy
The `load_models()` function implements a robust model loading mechanism:
- Searches for specific model files in the '../models' directory
- Supports multiple model types (Random Forest and SVM for regression and classification)
- Provides user feedback about model loading status via Streamlit sidebar

```python
def load_models():
    models = {}
    models_dir = os.path.abspath('../models')
    model_files = {
        'rf_reg': 'random_forest_regressor.joblib',
        'svm_reg': 'svm_regressor.joblib',
        'rf_clf': 'random_forest_classifier.joblib',
        'svm_clf': 'svm_classifier.joblib'
    }
    
    # Load each model type if file exists
    for key, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            models[key] = joblib.load(model_path)
    
    return models
```

## 3. Prediction Workflow

### Regression Prediction Process
1. User selects a ticker and input features via sidebar sliders
2. Calculates average sentiment score for the selected ticker
3. Scales input features using pre-trained scaler
4. Uses selected regression model for prediction
5. Provides recommendation based on price difference

Key steps in prediction:
```python
# Scale input data
input_data_scaled = scaler_features.transform(input_data)

# Predict using regression model
prediction_scaled = selected_model.predict(input_data_scaled)[0]

# Calculate price difference
current_price = ticker_unscaled[ticker_unscaled['Date'] == ticker_unscaled['Date'].max()]['Close'].values[0]
price_diff = prediction_scaled - current_price

# Determine recommendation
if price_diff > 0:
    recommendation = "BUY"
elif price_diff < 0:
    recommendation = "HOLD/SELL"
```

### Classification Prediction Process
Similar to regression, but predicts stock movement direction:
- Predicts whether stock will increase or decrease
- Provides prediction confidence percentage
- Visualizes prediction probabilities using a bar chart

## 4. Chatbot Integration

### Rasa Chatbot Communication
The `get_rasa_response()` function handles communication with a Rasa chatbot:
- Sends user messages to a local Rasa webhook
- Handles potential connection errors
- Returns chatbot responses

```python
def get_rasa_response(message_text):
    rasa_url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {"sender": "streamlit_user", "message": message_text}
    try:
        response = requests.post(rasa_url, json=payload, timeout=10)
        return response.json() if response.status_code == 200 else [{"text": "Error processing request"}]
    except Exception as e:
        return [{"text": f"Connection error: {str(e)}"}]
```

## 5. Data Visualization

### Interactive Visualizations
- Uses Plotly Express for creating dynamic charts
- Supports multiple visualization types:
  - Closing prices over time
  - Sentiment score distribution
  - Correlation heatmaps
  - Statistical distribution plots

Example visualization creation:
```python
fig_augmented = px.line()
for ticker in tickers:
    ticker_data = augmented_data_unscaled_sorted[augmented_data_unscaled_sorted['Ticker'] == ticker]
    fig_augmented.add_scatter(x=ticker_data['Date'], y=ticker_data['Close'], mode='lines', name=ticker)
```

# Implementation of Requests in Data Processing

## 1. Data Freshness and Retrieval Requests

### `is_file_fresh()` Method
The `is_file_fresh()` method is crucial for implementing intelligent data refresh requests. It determines whether existing data files are recent enough to be reused.

#### Key Implementation Details:
- Checks file modification time against a specified freshness threshold (default: 24 hours)
- Returns a boolean indicating whether the file is considered "fresh"
- Prevents unnecessary data fetching and reduces computational overhead

### `load_and_prepare_data()` Method
This method orchestrates the entire data loading and preparation process, handling complex request scenarios:

#### Request Handling Workflow:
1. **Freshness Check**:
   - Evaluates existing data files for currency
   - Determines if new data needs to be fetched
   - Supports manual refresh triggers

2. **Conditional Data Fetching**:
   - Triggers `fetch_stock_data()` and `fetch_sentiment_data()` methods only when necessary
   - Allows flexible data retrieval strategies

3. **Data Processing Pipeline**:
   - Cleans stock data
   - Applies feature scaling
   - Augments data with synthetic entries
   - Processes sentiment data

## 2. Stock Data Retrieval Requests: `fetch_stock_data()`

### API and Data Source Integration
- Utilizes `yfinance` library for retrieving historical stock data
- Supports multiple stock tickers simultaneously
- Implements robust error handling and logging

#### Key Request Handling Features:
- Flexible date range selection (default: approximately 5 years)
- Handles missing data gracefully
- Generates synthetic data for columns like 'Dividends' and 'Stock Splits' if unavailable

## 3. Sentiment Data Retrieval Requests: `fetch_sentiment_data()`

### Reddit API Integration
- Uses PRAW (Python Reddit API Wrapper) for fetching Reddit post data
- Performs sentiment analysis on post titles
- Securely manages API credentials through environment variables

#### Sentiment Analysis Workflow:
1. Initialize Reddit API client
2. Search for posts related to specific stock tickers
3. Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)
4. Generate sentiment scores

## 4. Data Augmentation Requests: `augment_with_fake_data()`

### Synthetic Data Generation Strategy
- Creates realistic fake stock data based on real data distributions
- Uses `Faker` library for generating diverse synthetic entries
- Applies standard scaling to maintain data consistency

#### Augmentation Process:
1. Generate fake entries matching original data characteristics
2. Combine real and synthetic data
3. Apply feature scaling
4. Save augmented dataset and scalers

## 5. Machine Learning Model Training Requests: `train_models()`

### Comprehensive Model Training Pipeline
- Supports both regression and classification model training
- Integrates sentiment scores as additional features
- Trains multiple model types (Random Forest, Support Vector Machines)

#### Model Training Workflow:
1. Prepare and merge stock and sentiment data
2. Create regression and classification targets
3. Split data into training and testing sets
4. Train multiple models
5. Evaluate model performance
6. Save models and performance metrics

## 6. Error Handling and Logging

### Robust Request Management
- Implements comprehensive logging throughout the request lifecycle
- Provides detailed error messages and diagnostic information
- Supports graceful failure and recovery mechanisms

# Work Done

I created and completed this project solo.


## Project Structure
```
stock-sentiment-recommender/
│
├── app/
│   ├── main.py            # Streamlit main application
│   ├── train_models.py
│   └── utils/
│       └── data_processing.py
│
├── data/
│   ├── raw/
│   │   ├── stock_data.csv
│   │   └── reddit_sentiment.csv
│   └── processed/
│       ├── augmented_stock_data.csv
│       └── cleaned_stock_data.csv
│
├── rasa_bot/
│   ├── config.yml
│   ├── domain.yml
│   ├── data/
│   │   ├── nlu.yml
│   │   └── stories.yml
│   └── actions/
│       └── actions.py
│
├── .gitignore
├── README.md
├── requirements.txt
```








## Contact
Huraira Ali

Email: csixh2welveosix@proton.me

GitLab: mygit.th-deg.de/ha06705