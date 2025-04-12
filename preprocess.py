import os
from datetime import timedelta, datetime
import polars as pl
from multiprocessing import Pool, cpu_count
from sentiment_analysis import sentimentanalysis  # Import the sentiment analysis function

def convert_to_utc(time_str):
    if isinstance(time_str, float) or not isinstance(time_str, str):
        return "Invalid date format"
        
    if " EDT" in time_str:
        time_str_cleaned = time_str.replace(" EDT", "")
        offset = timedelta(hours=-4)
    elif " EST" in time_str:
        time_str_cleaned = time_str.replace(" EST", "")
        offset = timedelta(hours=-5)
    else:
        offset = timedelta(hours=0)
        time_str_cleaned = time_str

    formats = [
        '%B %d, %Y — %I:%M %p',  # e.g., "September 12, 2023 — 06:15 pm"
        '%b %d, %Y %I:%M%p',      # e.g., "Nov 14, 2023 7:35AM"
        '%d-%b-%y',              # e.g., "6-Jan-22"
        '%Y-%m-%d',              # e.g., "2021-4-5"
        '%Y/%m/%d',              # e.g., "2021/4/5"
        '%b %d, %Y',             # e.g., "DEC 7, 2023"
        '%Y-%m-%d %H:%M:%S'      # e.g., "2023-12-07 16:30:00"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str_cleaned, fmt)
            # For the '%d-%b-%y' format, ignore offset adjustment
            if fmt == '%d-%b-%y':
                offset = timedelta(hours=0)
            dt_utc = dt + offset
            return dt_utc.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue

    return "Invalid date format"

def process_nasdaq_data_parquet(file_path, saving_path):
    print('Processing NASDAQ news data...')
    try:
        df = pl.scan_csv(file_path)
        
        if "Unnamed: 0" in df.collect_schema().names():
            df = df.drop("Unnamed: 0")
        
        # Convert the Date column to UTC
        date_df = df.select("Date").collect()
        converted_dates = date_df.with_columns([
            pl.col("Date").map_elements(convert_to_utc, return_dtype=pl.Utf8).alias("Date")
        ])
        
        # Replace the original Date column with the converted dates
        df = df.with_columns([
            pl.lit(converted_dates.get_column("Date")).alias("Date")
        ])
        
        # Sort by the Date string (ISO-formatted strings sort chronologically)
        df = df.sort("Date", descending=True)
        
        # Ensure saving path exists
        os.makedirs(saving_path, exist_ok=True)
        output_file = os.path.join(saving_path, os.path.basename(file_path).replace('.csv', '.parquet'))
        
        # Execute the lazy query and write directly to parquet
        df.collect().write_parquet(output_file)
        print('Done processing NASDAQ news data. Saved to:', output_file)
        
    except Exception as e:
        print(f"Error processing NASDAQ news data: {str(e)}")
        raise 

def process_single_stock_file(args):
    file_path, saving_path = args
    csv_file = os.path.basename(file_path)
    print(f'Processing {csv_file}...')
    
    try:
        df = pl.read_csv(file_path)

        # Lowercase all column names
        for col in df.columns:
            df = df.rename({col: col.lower()})
        
        # Identify and standardize the date column (rename to "Date")
        date_columns = ["datetime", "date", "time", "timestamp"]
        for col in date_columns:
            if col.lower() in df.columns:
                df = df.rename({col: "Date"})
                break
        
        if "Date" not in df.columns:
            print(f"No date column found in {csv_file}, skipping...")
            return
        
        # Convert date to UTC
        df = df.with_columns([
            pl.col("Date").map_elements(convert_to_utc, return_dtype=pl.Utf8).alias("Date")
        ])
        df = df.with_columns([
            pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", utc=True)
        ])
        
        # Sort by date descending
        df = df.sort("Date", descending=True)
        
        output_file = os.path.join(saving_path, csv_file.replace('.csv', '.parquet'))
        df.write_parquet(output_file)
        print(f'Done processing {csv_file}. Saved to: {output_file}')
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

def process_stock_history_parquet(folder_path, saving_path):
    print('Processing full stock price history...')
    os.makedirs(saving_path, exist_ok=True)
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_paths = [(os.path.join(folder_path, f), saving_path) for f in csv_files]
    
    num_processes = max(1, cpu_count() - 1)
    print(f'Using {num_processes} processes for parallel processing...')
    
    with Pool(processes=num_processes) as pool:
        pool.map(process_single_stock_file, file_paths)
    
    print('Completed processing all stock history files.')

def process_sentiment_analysis(input_file, saving_path):
    print('Processing sentiment analysis...')
    try:
        # Read the preprocessed news data
        news_data = pl.read_parquet(input_file)
        
        # Create lists to store sentiment results
        sentiment_scores = []
        sentiment_magnitudes = []
        
        # Process each article
        for row in news_data.iter_rows():
            article = row[news_data.columns.index('Article')]
            if isinstance(article, str) and len(article.strip()) > 0:
                try:
                    sentiment_score, sentiment_magnitude = sentimentanalysis(article)
                except Exception as e:
                    print(f"Error processing article: {e}")
                    sentiment_score, sentiment_magnitude = None, None
            else:
                sentiment_score, sentiment_magnitude = None, None
                
            sentiment_scores.append(sentiment_score)
            sentiment_magnitudes.append(sentiment_magnitude)
        
        # Create a new dataframe with sentiment results
        sentiment_df = pl.DataFrame({
            'Date': news_data['Date'],
            'Title': news_data['Title'],
            'Article': news_data['Article'],
            'Sentiment_Score': sentiment_scores,
            'Sentiment_Magnitude': sentiment_magnitudes
        })
        
        # Sort by date descending
        sentiment_df = sentiment_df.sort('Date', descending=True)
        
        # Ensure saving path exists
        os.makedirs(saving_path, exist_ok=True)
        
        # Create output filename based on input filename
        base_name = os.path.basename(input_file)
        output_file = os.path.join(saving_path, f"sentiment_{base_name}")
        
        # Save to parquet
        sentiment_df.write_parquet(output_file)
        print(f'Done processing sentiment analysis. Saved to: {output_file}')
        
    except Exception as e:
        print(f"Error processing sentiment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Define paths for the input files and output folders
    # For NASDAQ news data:
    nasdaq_file_path = 'nasdaq_external_data.csv'
    nasdaq_saving_path = 'nasdaq_data_preprocessed'
    
    # For full stock price history:
    full_history_folder_path = 'full_history'
    full_history_saving_path = 'full_history_preprocessed'
    
    # For sentiment analysis:
    sentiment_saving_path = 'sentiment_data'
    
    # Process the NASDAQ news data and save as Parquet
    nasdaq_parquet_path = os.path.join(nasdaq_saving_path, os.path.basename(nasdaq_file_path).replace('.csv', '.parquet'))
    process_nasdaq_data_parquet(nasdaq_file_path, nasdaq_saving_path)
    
    # Process sentiment analysis on the NASDAQ news data
    process_sentiment_analysis(nasdaq_parquet_path, sentiment_saving_path)
    
    # Process the full stock history and save each file as Parquet
    process_stock_history_parquet(full_history_folder_path, full_history_saving_path)
