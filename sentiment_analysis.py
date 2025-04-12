import os
import polars as pl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# Download the VADER lexicon if not already available
nltk.download('vader_lexicon')

def process_chunk(chunk):
    """
    Process a chunk of articles, computing sentiment scores using VADER.
    This function reinitializes the SentimentIntensityAnalyzer in each process.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia_local = SentimentIntensityAnalyzer()
    results = []
    for article in chunk:
        if not isinstance(article, str) or len(article.strip()) == 0:
            results.append({"compound": None, "pos": None, "neg": None, "neu": None})
        else:
            results.append(sia_local.polarity_scores(article))
    return results

def process_sentiment(input_parquet, output_parquet, text_column="Article"):
    """
    Loads the preprocessed NASDAQ news data from a Parquet file, performs sentiment analysis on the
    specified text column using parallel processing, and writes out a new Parquet file with added
    sentiment columns.
    """
    print(f"Loading preprocessed data from {input_parquet}...")
    df = pl.read_parquet(input_parquet)
    n_rows = df.shape[0]
    print(f"Loaded {n_rows} rows.")
    
    # Extract the text column as a list
    articles = df[text_column].to_list()
    
    # Determine number of workers and chunk size based on available CPUs
    n_workers = os.cpu_count() or 4
    chunk_size = math.ceil(len(articles) / n_workers)
    chunks = [articles[i:i + chunk_size] for i in range(0, len(articles), chunk_size)]
    
    print(f"Processing {len(articles)} articles in {len(chunks)} chunks using {n_workers} workers...")
    
    # Use ProcessPoolExecutor to process chunks in parallel
    sentiment_scores = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            sentiment_scores.extend(future.result())
    
    # Extract sentiment score lists
    compound_scores = [score["compound"] for score in sentiment_scores]
    pos_scores = [score["pos"] for score in sentiment_scores]
    neg_scores = [score["neg"] for score in sentiment_scores]
    neu_scores = [score["neu"] for score in sentiment_scores]
    
    # Add sentiment columns to the DataFrame
    df = df.with_columns([
        pl.Series("compound", compound_scores),
        pl.Series("pos", pos_scores),
        pl.Series("neg", neg_scores),
        pl.Series("neu", neu_scores)
    ])
        
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    
    print(f"Saving sentiment analysis output to {output_parquet}...")
    df.write_parquet(output_parquet)
    print("Sentiment analysis complete.")

if __name__ == "__main__":
    input_parquet = "nasdaq_data_preprocessed/nasdaq_external_data.parquet"
    output_parquet = "nasdaq_data_preprocessed/nasdaq_external_data_sentiment.parquet"
    
    process_sentiment(input_parquet, output_parquet)
