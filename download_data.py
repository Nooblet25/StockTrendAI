import os
import kaggle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def download_dataset():
    """
    Download the Yahoo Finance dataset from Kaggle
    """
    print("Downloading dataset from Kaggle...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download the dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update',
            path=data_dir,
            unzip=True
        )
        print("Dataset downloaded successfully!")
        
        # Process the downloaded files
        process_dataset(data_dir)
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nTo use the Kaggle API, you need to:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click on 'Create New API Token'")
        print("3. Download the kaggle.json file")
        print("4. Create a .kaggle directory in your home folder")
        print("5. Move the kaggle.json file to ~/.kaggle/")
        print("6. Run: chmod 600 ~/.kaggle/kaggle.json")

def process_dataset(data_dir):
    """
    Process the downloaded dataset files
    """
    print("\nProcessing dataset files...")
    
    # List all CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in the data directory!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each file with a progress bar
    for csv_file in tqdm(csv_files, desc="Processing files"):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Ensure required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                tqdm.write(f"Skipping {csv_file.name} - Missing required columns")
                continue
            
            # Convert date column with UTC=True to handle timezone warnings
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Save processed file
            output_file = data_dir / csv_file.name
            df.to_csv(output_file, index=False)
            tqdm.write(f"Processed {csv_file.name} - {len(df)} rows")
            
        except Exception as e:
            tqdm.write(f"Error processing {csv_file.name}: {str(e)}")

if __name__ == "__main__":
    download_dataset() 