import pandas as pd
import numpy as np
import requests
import logging
import os
from typing import Dict, List, Union, Optional
from datetime import datetime, timedelta
import time
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Class for retrieving and analyzing sentiment data from various sources
    including social media (Twitter/X, Reddit) and news APIs.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the sentiment analyzer with API keys and configuration
        
        Args:
            config: Dictionary containing API keys and configuration settings
        """
        self.config = config or {}
        self.news_api_key = self.config.get('news_api_key', os.getenv('NEWS_API_KEY'))
        self.twitter_bearer_token = self.config.get('twitter_bearer_token', os.getenv('TWITTER_BEARER_TOKEN'))
        self.reddit_client_id = self.config.get('reddit_client_id', os.getenv('REDDIT_CLIENT_ID'))
        self.reddit_client_secret = self.config.get('reddit_client_secret', os.getenv('REDDIT_CLIENT_SECRET'))
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize transformers sentiment pipeline if deep learning is enabled
        self.use_deep_learning = self.config.get('use_deep_learning', False)
        if self.use_deep_learning:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True,
                    max_length=512
                )
            except Exception as e:
                logger.warning(f"Could not load deep learning sentiment model: {e}")
                self.use_deep_learning = False
    
    def get_sentiment_for_period(self, 
                                symbol: str, 
                                start_time: datetime, 
                                end_time: datetime) -> pd.DataFrame:
        """
        Get aggregated sentiment data for a specific cryptocurrency over a time period
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            start_time: Start datetime for sentiment analysis
            end_time: End datetime for sentiment analysis
            
        Returns:
            DataFrame with sentiment features indexed by timestamp
        """
        # Convert symbol to common search terms
        search_terms = self._get_search_terms(symbol)
        
        # Fetch data from different sources in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            if self.news_api_key:
                futures.append(executor.submit(self.get_news_sentiment, search_terms, start_time, end_time))
            
            if self.twitter_bearer_token:
                futures.append(executor.submit(self.get_twitter_sentiment, search_terms, start_time, end_time))
                
            if self.reddit_client_id and self.reddit_client_secret:
                futures.append(executor.submit(self.get_reddit_sentiment, search_terms, start_time, end_time))
            
            # Collect results
            sentiment_dfs = []
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        sentiment_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error collecting sentiment data: {e}")
        
        # If no data was collected, return empty dataframe with expected columns
        if not sentiment_dfs:
            logger.warning(f"No sentiment data collected for {symbol}")
            return pd.DataFrame(columns=['timestamp', 'sentiment_score', 'sentiment_magnitude', 
                                        'positive_ratio', 'negative_ratio', 'neutral_ratio'])
        
        # Combine and aggregate results
        combined_df = pd.concat(sentiment_dfs, ignore_index=True)
        
        # Resample to hourly data (or other interval based on config)
        interval = self.config.get('sentiment_interval', '1H')
        
        # Ensure timestamp is datetime
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df.set_index('timestamp', inplace=True)
        
        # Aggregate by time interval
        aggregated = combined_df.resample(interval).agg({
            'sentiment_score': 'mean',
            'sentiment_magnitude': 'mean',
            'positive_ratio': 'mean',
            'negative_ratio': 'mean',
            'neutral_ratio': 'mean'
        }).reset_index()
        
        # Fill missing values with neutral sentiment
        aggregated.fillna({
            'sentiment_score': 0,
            'sentiment_magnitude': 0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34
        }, inplace=True)
        
        return aggregated
    
    def get_news_sentiment(self, 
                          search_terms: List[str], 
                          start_time: datetime, 
                          end_time: datetime) -> pd.DataFrame:
        """
        Fetch and analyze sentiment from news articles
        
        Args:
            search_terms: List of terms to search for
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with sentiment data
        """
        if not self.news_api_key:
            logger.warning("News API key not provided")
            return pd.DataFrame()
            
        results = []
        
        for term in search_terms:
            try:
                # Format dates for News API
                from_date = start_time.strftime('%Y-%m-%d')
                to_date = end_time.strftime('%Y-%m-%d')
                
                # Call News API
                url = f"https://newsapi.org/v2/everything"
                params = {
                    "q": term,
                    "from": from_date,
                    "to": to_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "apiKey": self.news_api_key
                }
                
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    logger.error(f"News API error: {response.text}")
                    continue
                    
                data = response.json()
                
                if data.get("status") != "ok":
                    logger.error(f"News API returned error: {data}")
                    continue
                
                articles = data.get("articles", [])
                
                # Process each article
                for article in articles:
                    published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = article.get('content', '')
                    
                    # Combine text for sentiment analysis
                    text = f"{title} {description} {content}"
                    
                    # Get sentiment
                    sentiment = self._analyze_text_sentiment(text)
                    sentiment['timestamp'] = published_at
                    sentiment['source'] = 'news'
                    sentiment['search_term'] = term
                    
                    results.append(sentiment)
                    
            except Exception as e:
                logger.error(f"Error fetching news sentiment for {term}: {str(e)}")
        
        # Create DataFrame
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results)
    
    def get_twitter_sentiment(self, 
                             search_terms: List[str], 
                             start_time: datetime, 
                             end_time: datetime) -> pd.DataFrame:
        """
        Fetch and analyze sentiment from Twitter/X
        
        Args:
            search_terms: List of terms to search for
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with sentiment data
        """
        if not self.twitter_bearer_token:
            logger.warning("Twitter bearer token not provided")
            return pd.DataFrame()
            
        results = []
        
        for term in search_terms:
            try:
                # Format dates for Twitter API v2
                start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Call Twitter API v2 recent search endpoint
                url = "https://api.twitter.com/2/tweets/search/recent"
                headers = {
                    "Authorization": f"Bearer {self.twitter_bearer_token}"
                }
                params = {
                    "query": f"{term} -is:retweet lang:en",
                    "start_time": start_str,
                    "end_time": end_str,
                    "max_results": 100,
                    "tweet.fields": "created_at,public_metrics"
                }
                
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Twitter API error: {response.status_code}, {response.text}")
                    continue
                
                data = response.json()
                tweets = data.get("data", [])
                
                # Process each tweet
                for tweet in tweets:
                    created_at = datetime.strptime(tweet['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')
                    text = tweet.get('text', '')
                    
                    # Get sentiment
                    sentiment = self._analyze_text_sentiment(text)
                    
                    # Add engagement metrics if available
                    metrics = tweet.get('public_metrics', {})
                    engagement = metrics.get('like_count', 0) + metrics.get('retweet_count', 0)
                    sentiment['engagement'] = engagement
                    sentiment['timestamp'] = created_at
                    sentiment['source'] = 'twitter'
                    sentiment['search_term'] = term
                    
                    results.append(sentiment)
                    
            except Exception as e:
                logger.error(f"Error fetching Twitter sentiment for {term}: {str(e)}")
        
        # Create DataFrame
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results)
    
    def get_reddit_sentiment(self, 
                            search_terms: List[str], 
                            start_time: datetime, 
                            end_time: datetime) -> pd.DataFrame:
        """
        Fetch and analyze sentiment from Reddit
        
        Args:
            search_terms: List of terms to search for
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with sentiment data
        """
        if not (self.reddit_client_id and self.reddit_client_secret):
            logger.warning("Reddit API credentials not provided")
            return pd.DataFrame()
            
        results = []
        
        # Get Reddit OAuth token
        try:
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials"
            }
            auth_headers = {
                "User-Agent": "CryptoVisionAI/1.0"
            }
            
            response = requests.post(
                auth_url,
                data=auth_data,
                auth=(self.reddit_client_id, self.reddit_client_secret),
                headers=auth_headers
            )
            
            if response.status_code != 200:
                logger.error(f"Reddit auth error: {response.status_code}, {response.text}")
                return pd.DataFrame()
            
            token_data = response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                logger.error("Failed to get Reddit access token")
                return pd.DataFrame()
                
            # Set up headers for API calls
            api_headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "CryptoVisionAI/1.0"
            }
            
            # Convert timestamps to Reddit format (Unix timestamps)
            start_unix = int(start_time.timestamp())
            end_unix = int(end_time.timestamp())
            
            for term in search_terms:
                # Query Reddit API for submissions
                url = f"https://oauth.reddit.com/search"
                params = {
                    "q": term,
                    "sort": "new",
                    "t": "all",
                    "limit": 100
                }
                
                response = requests.get(url, headers=api_headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Reddit API error: {response.status_code}, {response.text}")
                    continue
                    
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                
                # Process each post
                for post in posts:
                    post_data = post.get("data", {})
                    created_utc = post_data.get("created_utc", 0)
                    
                    # Check if within time range
                    if not (start_unix <= created_utc <= end_unix):
                        continue
                    
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    created_at = datetime.fromtimestamp(created_utc)
                    
                    # Combine text for sentiment analysis
                    text = f"{title} {selftext}"
                    if not text.strip():
                        continue
                    
                    # Get sentiment
                    sentiment = self._analyze_text_sentiment(text)
                    
                    # Add engagement metrics if available
                    score = post_data.get("score", 0)
                    num_comments = post_data.get("num_comments", 0)
                    engagement = score + num_comments
                    
                    sentiment['engagement'] = engagement
                    sentiment['timestamp'] = created_at
                    sentiment['source'] = 'reddit'
                    sentiment['search_term'] = term
                    
                    results.append(sentiment)
                    
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment: {str(e)}")
        
        # Create DataFrame
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results)
    
    def adaptive_sentiment_weighting(self, sentiment_data: pd.DataFrame, price_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply adaptive weighting to sentiment data based on historical correlation with price movements
        
        Args:
            sentiment_data (pd.DataFrame): DataFrame containing sentiment metrics
            price_data (pd.DataFrame, optional): OHLCV price data for correlation analysis
            
        Returns:
            pd.DataFrame: Weighted sentiment data
        """
        weighted_data = sentiment_data.copy()
        
        # If no price data provided, return original sentiment data
        if price_data is None or price_data.empty:
            logger.warning("No price data provided for adaptive sentiment weighting")
            return weighted_data
            
        try:
            # Calculate base weights using exponential decay (more recent = more important)
            time_weights = np.exp(-0.1 * np.arange(len(sentiment_data)))
            time_weights = time_weights / np.sum(time_weights)  # Normalize
            
            # Calculate historical correlation between sentiment and future price movements
            if len(price_data) >= len(sentiment_data):
                # Align price data with sentiment data
                aligned_price = price_data.reindex(sentiment_data.index)
                
                # Get future 24h returns
                future_returns = aligned_price['close'].pct_change(24).shift(-24)
                
                # Calculate correlation for each sentiment column
                correlations = {}
                for col in sentiment_data.columns:
                    if col in ['compound', 'positive', 'negative', 'sentiment_score', 'sentiment_magnitude']:
                        # Use absolute correlation as weight (correlation strength matters)
                        corr = abs(sentiment_data[col].corr(future_returns))
                        if not np.isnan(corr):
                            correlations[col] = max(0.1, corr)  # Minimum weight of 0.1
                        else:
                            correlations[col] = 0.5  # Default weight if correlation is NaN
                
                # Normalize correlation weights
                total_corr = sum(correlations.values())
                if total_corr > 0:
                    correlations = {k: v/total_corr for k, v in correlations.items()}
                    
                # Apply both time weights and correlation weights to sentiment data
                for col, weight in correlations.items():
                    weighted_data[col] = sentiment_data[col] * weight
                    
                logger.info(f"Applied adaptive sentiment weighting with correlations: {correlations}")
            
        except Exception as e:
            logger.warning(f"Error in adaptive sentiment weighting: {str(e)}")
            logger.warning("Using original sentiment data without weighting")
            
        return weighted_data
    
    def get_aggregated_sentiment(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get aggregated sentiment data for a specific symbol and time period
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_date (datetime): Start of time period
            end_date (datetime): End of time period
            
        Returns:
            pd.DataFrame: Aggregated sentiment data
        """
        # Get raw sentiment data
        sentiment_data = self.get_sentiment_for_period(symbol, start_date, end_date)
        
        # Apply adaptive weighting if price data is available
        try:
            from src.data_processing.binance_connector import BinanceConnector
            
            # Get historical price data for the same period
            connector = BinanceConnector()
            price_data = connector.get_historical_data(
                symbol=symbol,
                interval='1h',
                start_time=start_date,
                end_time=end_date
            )
            
            # Apply adaptive weighting using price data
            if not price_data.empty:
                sentiment_data = self.adaptive_sentiment_weighting(sentiment_data, price_data)
                
        except Exception as e:
            logger.warning(f"Could not apply adaptive sentiment weighting: {str(e)}")
            logger.warning("Using unweighted sentiment data")
        
        return sentiment_data
    
    def _analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a text using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment metrics
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment_score': 0,
                'sentiment_magnitude': 0,
                'positive_ratio': 0.33,
                'negative_ratio': 0.33,
                'neutral_ratio': 0.34
            }
        
        # Clean text
        text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores['compound']
        pos = vader_scores['pos']
        neg = vader_scores['neg']
        neu = vader_scores['neu']
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combine scores (weighted average)
        sentiment_score = (compound + textblob_polarity) / 2
        sentiment_magnitude = textblob_subjectivity
        
        # Use deep learning model if enabled (override previous scores)
        if self.use_deep_learning and hasattr(self, 'sentiment_pipeline'):
            try:
                result = self.sentiment_pipeline(text[:512])[0]
                label = result['label']
                score = result['score']
                
                if label == "POSITIVE":
                    dl_score = score
                elif label == "NEGATIVE":
                    dl_score = -score
                else:
                    dl_score = 0
                    
                # Use deep learning score with more weight
                sentiment_score = (sentiment_score + 2 * dl_score) / 3
                
            except Exception as e:
                logger.warning(f"Error in deep learning sentiment analysis: {str(e)}")
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_magnitude': sentiment_magnitude,
            'positive_ratio': pos,
            'negative_ratio': neg,
            'neutral_ratio': neu
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_search_terms(self, symbol: str) -> List[str]:
        """
        Convert cryptocurrency symbol to common search terms
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")
            
        Returns:
            List of search terms
        """
        search_terms = []
        
        # Add common variations
        if symbol.upper() == "BTC":
            search_terms = ["Bitcoin", "BTC", "#Bitcoin", "#BTC", "Bitcoin price"]
        elif symbol.upper() == "ETH":
            search_terms = ["Ethereum", "ETH", "#Ethereum", "#ETH", "Ethereum price"]
        elif symbol.upper() == "SOL":
            search_terms = ["Solana", "SOL", "#Solana", "#SOL", "Solana price"]
        elif symbol.upper() == "XRP":
            search_terms = ["Ripple", "XRP", "#Ripple", "#XRP", "Ripple price"]
        elif symbol.upper() == "ADA":
            search_terms = ["Cardano", "ADA", "#Cardano", "#ADA", "Cardano price"]
        elif symbol.upper() == "DOT":
            search_terms = ["Polkadot", "DOT", "#Polkadot", "#DOT", "Polkadot price"]
        else:
            # Generic terms
            search_terms = [f"{symbol}", f"#{symbol}", f"{symbol} price"]
            
        return search_terms