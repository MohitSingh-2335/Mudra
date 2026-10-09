# prepare_data.py

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from config import BTCUSDT_1H_CSV, FEATURED_BTC_DATA_PATH
import os

def preprocess_and_save_featured_data(input_path=BTCUSDT_1H_CSV, output_path=FEATURED_BTC_DATA_PATH):
    """
    Loads raw data, engineers all features, and saves the result to a new CSV file.
    """
    print("Starting data preparation...")

    # Load and clean the initial data
    df = load_and_clean_data(input_path)

    df = merge_fear_greed(df, timestamp_col='timestamp')

    # Merge on-chain metrics (fails soft to skipping features if API is down)
    df = merge_onchain(df, timestamp_col='timestamp')
    
    # Engineer all features
    featured_df = create_features(df)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the new DataFrame with all features
    featured_df.to_csv(output_path, index=False)
    
    print(f"Data preparation complete. Featured data saved to '{output_path}'.")

if __name__ == '__main__':
    preprocess_and_save_featured_data()