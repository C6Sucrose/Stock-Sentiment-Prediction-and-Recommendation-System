# app/train_models.py

from utils.data_processing import load_and_prepare_data, train_models

def main():
    # Load and prepare data
    cleaned_stock_data_scaled, augmented_data_scaled, sentiment_data, cleaned_stock_data_unscaled, augmented_data_unscaled = load_and_prepare_data(fetch=True, manual_refresh=False)

    # Train and save models
    train_models(augmented_data_scaled, sentiment_data)

if __name__ == "__main__":
    main()