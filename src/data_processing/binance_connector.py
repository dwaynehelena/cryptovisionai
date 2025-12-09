#!/usr/bin/env python3
"""
Binance Connector Module - Handles connections to Binance API for real-time and historical data
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import os
import yaml
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Configure logging
logger = logging.getLogger("binance_connector")

class BinanceConnector:
    """
    Handles connections to Binance API for fetching real-time
    and historical price data, market information and executing trades
    """
    
    def __init__(self, use_testnet: bool = True, config_path: str = "../config/config.yaml", 
                 api_key: str = None, api_secret: str = None):
        """
        Initialize the Binance connector
        
        Args:
            use_testnet (bool): Whether to use the testnet
            config_path (str): Path to configuration file
            api_key (str): Binance API key (optional, overrides config file)
            api_secret (str): Binance API secret (optional, overrides config file)
        """
        self.use_testnet = use_testnet
        self.config_path = config_path
        
        # Load config from file if it exists, otherwise use empty dict
        try:
            self.config = self._load_config(config_path)
        except Exception:
            logger.warning(f"Could not load config from {config_path}, relying on provided keys")
            self.config = {}
        
        # Initialize the client
        if use_testnet:
            if not api_key or not api_secret:
                # First try api.binance.testnet path (nested structure)
                try:
                    api_key = self.config['api']['binance']['testnet']['api_key']
                    api_secret = self.config['api']['binance']['testnet']['api_secret']
                except (KeyError, TypeError):
                    # Fall back to binance.api_key path (flat structure)
                    try:
                        api_key = self.config['binance']['api_key']
                        api_secret = self.config['binance']['api_secret']
                    except (KeyError, TypeError):
                        pass
                
            if api_key and api_secret:
                # Strip whitespace to avoid format errors
                api_key = str(api_key).strip()
                api_secret = str(api_secret).strip()
                self.client = Client(api_key, api_secret, testnet=True)
                logger.info("Initialized Binance connector using testnet")
            else:
                logger.error("Failed to initialize Binance connector: Missing API keys for testnet")
                self.client = None
        else:
            if not api_key or not api_secret:
                # First try api.binance.mainnet path (nested structure)
                try:
                    api_key = self.config['api']['binance']['mainnet']['api_key']
                    api_secret = self.config['api']['binance']['mainnet']['api_secret']
                except (KeyError, TypeError):
                    # Fall back to binance.api_key path (flat structure)
                    try:
                        api_key = self.config['binance']['api_key']
                        api_secret = self.config['binance']['api_secret']
                    except (KeyError, TypeError):
                        pass
                
            if api_key and api_secret:
                # Strip whitespace to avoid format errors
                api_key = str(api_key).strip()
                api_secret = str(api_secret).strip()
                self.client = Client(api_key, api_secret)
                logger.info("Initialized Binance connector using mainnet")
            else:
                logger.warning("Missing API keys for mainnet. Initializing in public-only mode.")
                self.client = Client(None, None)
        
        # Cache for server time offset
        self.server_time_offset = 0
        self._update_server_time_offset()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _update_server_time_offset(self) -> None:
        """Update the server time offset to ensure accurate timestamps"""
        try:
            # Get server time
            server_time = self.client.get_server_time()
            server_timestamp = server_time['serverTime']
            local_timestamp = int(time.time() * 1000)
            self.server_time_offset = server_timestamp - local_timestamp
            logger.debug(f"Updated server time offset: {self.server_time_offset} ms")
        except Exception as e:
            logger.error(f"Error updating server time offset: {e}")
    
    def calculate_latency(self) -> float:
        """
        Calculate API latency
        
        Returns:
            float: Latency in milliseconds
        """
        try:
            start_time = time.time()
            self.client.get_server_time()
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            return latency
        except Exception as e:
            logger.error(f"Error calculating latency: {e}")
            return -1

    def get_active_symbols(self, base_asset: str = None, quote_asset: str = "USDT") -> List[str]:
        """
        Get list of active trading symbols
        
        Args:
            base_asset (str): Filter by base asset (e.g. 'BTC')
            quote_asset (str): Filter by quote asset (e.g. 'USDT')
            
        Returns:
            List[str]: List of symbol names
        """
        try:
            exchange_info = self.client.get_exchange_info()
            
            # Filter only active symbols
            active_symbols = [
                s['symbol'] for s in exchange_info['symbols'] 
                if s['status'] == 'TRADING'
            ]
            
            # Apply filters
            if base_asset:
                active_symbols = [s for s in active_symbols if s.startswith(base_asset)]
                
            if quote_asset:
                active_symbols = [s for s in active_symbols if s.endswith(quote_asset)]
            
            return active_symbols
            
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker price for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: Ticker information
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_account_info(self) -> Dict:
        """
        Get account information including balances
        
        Returns:
            Dict: Account information
        """
        try:
            account_info = self.client.get_account()
            return account_info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'error': str(e)}
    
    def get_exchange_info(self) -> Dict:
        """
        Get exchange information including symbol details
        
        Returns:
            Dict: Exchange information
        """
        try:
            exchange_info = self.client.get_exchange_info()
            return exchange_info
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {'error': str(e)}
    
    def get_historical_klines(self, 
                             symbol: str, 
                             interval: str, 
                             limit: int = 500,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Get historical klines (candlestick data)
        
        Args:
            symbol (str): Trading pair symbol
            interval (str): Kline interval (e.g. '1m', '1h', '1d')
            limit (int): Number of klines to retrieve (max 1000)
            start_time (datetime): Start time
            end_time (datetime): End time
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            # Convert datetime to milliseconds if provided
            start_str = None
            end_str = None
            
            if start_time:
                start_str = int(start_time.timestamp() * 1000)
                
            if end_time:
                end_str = int(end_time.timestamp() * 1000)
            
            logger.info(f"Fetching klines for {symbol} {interval} from {start_str} to {end_str}")
            
            # Get klines from Binance using the generator for pagination
            # If limit is specified but no start time, we need to handle it carefully
            # The library's get_historical_klines fetches ALL data between start and end
            
            if start_str is None and limit > 1000:
                # If we want more than 1000 candles but didn't specify start time,
                # we should calculate a start time to avoid fetching everything
                # Estimate start time based on interval
                # This is a rough estimation
                if interval.endswith('m'):
                    minutes = int(interval[:-1])
                    delta = timedelta(minutes=minutes * limit)
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    delta = timedelta(hours=hours * limit)
                elif interval.endswith('d'):
                    days = int(interval[:-1])
                    delta = timedelta(days=days * limit)
                else:
                    delta = timedelta(days=limit) # Fallback
                
                start_dt = datetime.now() - delta
                start_str = int(start_dt.timestamp() * 1000)
                logger.info(f"Calculated start time: {start_str}")
            
            # Manual pagination to ensure we get all data
            klines = []
            current_start = start_str
            
            while True:
                # Fetch batch
                batch = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    startTime=current_start,
                    endTime=end_str
                )
                
                if not batch:
                    break
                    
                klines.extend(batch)
                logger.info(f"Fetched batch of {len(batch)} klines, total: {len(klines)}")
                
                # Update start time for next batch (last close time + 1ms)
                last_close_time = batch[-1][6]
                current_start = last_close_time + 1
                
                # If we got fewer than 1000, we reached the end
                if len(batch) < 1000:
                    break
                
                # Safety break if we have enough (if limit was specified and we exceeded it significantly)
                # But here we want ALL data since start_time, so we only break if we reach end_str
                if end_str and current_start > end_str:
                    break
            
            logger.info(f"Retrieved total {len(klines)} klines")
            
            if not klines:
                logger.warning(f"No klines returned for {symbol} {interval}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting klines for {symbol} {interval}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to a DataFrame
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        # Add SMA indicators
        for window in [7, 20, 50, 200]:
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
        
        # Add EMA indicators
        for window in [7, 20, 50, 200]:
            df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Add Bollinger Bands (20, 2)
        window = 20
        std_dev = 2
        df[f'BB_mid'] = df['close'].rolling(window=window).mean()
        df[f'BB_std'] = df['close'].rolling(window=window).std()
        df[f'BB_high'] = df[f'BB_mid'] + (df[f'BB_std'] * std_dev)
        df[f'BB_low'] = df[f'BB_mid'] - (df[f'BB_std'] * std_dev)
        
        # Add RSI (14)
        window = 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # Add MACD (12, 26, 9)
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Add Average True Range (ATR)
        window = 14
        tr1 = pd.DataFrame(df['high'] - df['low'])
        tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'ATR_{window}'] = tr.rolling(window=window).mean()
        
        # Add On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Depth of the order book
            
        Returns:
            Dict: Order book data
        """
        try:
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            return order_book
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of trades to retrieve
            
        Returns:
            List[Dict]: Recent trades
        """
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return [{'error': str(e)}]
    
    def get_historical_trades(self, symbol: str, limit: int = 500, from_id: Optional[int] = None) -> List[Dict]:
        """
        Get historical trades for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of trades to retrieve
            from_id (int): Trade ID to fetch from
            
        Returns:
            List[Dict]: Historical trades
        """
        try:
            trades = self.client.get_historical_trades(symbol=symbol, limit=limit, fromId=from_id)
            return trades
        except Exception as e:
            logger.error(f"Error getting historical trades for {symbol}: {e}")
            return [{'error': str(e)}]
    
    def get_aggregated_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get aggregated trades for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Number of trades to retrieve
            
        Returns:
            List[Dict]: Aggregated trades
        """
        try:
            trades = self.client.get_aggregate_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"Error getting aggregated trades for {symbol}: {e}")
            return [{'error': str(e)}]
    
    def create_order(self, 
                    symbol: str, 
                    side: str, 
                    order_type: str, 
                    quantity: float = None,
                    quoteOrderQty: float = None,
                    price: float = None,
                    timeInForce: str = 'GTC',
                    stopPrice: float = None,
                    **params) -> Dict:
        """
        Create an order on Binance
        
        Args:
            symbol (str): Trading pair symbol
            side (str): 'BUY' or 'SELL'
            order_type (str): 'LIMIT', 'MARKET', 'STOP_LOSS', etc.
            quantity (float): Order quantity
            quoteOrderQty (float): Quote order quantity (for MARKET orders)
            price (float): Order price (for LIMIT orders)
            timeInForce (str): Time in force
            stopPrice (float): Stop price (for stop orders)
            **params: Additional parameters
            
        Returns:
            Dict: Order information
        """
        try:
            if not self.use_testnet:
                # Double check before doing a real trade
                logger.warning(f"Creating REAL order: {side} {order_type} {quantity} {symbol}")
            
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                quoteOrderQty=quoteOrderQty,
                price=price,
                timeInForce=timeInForce,
                stopPrice=stopPrice,
                **params
            )
            
            logger.info(f"Created {side} {order_type} order for {quantity} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return {'error': str(e)}
    
    def get_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """
        Get order information
        
        Args:
            symbol (str): Trading pair symbol
            order_id (int): Order ID
            orig_client_order_id (str): Original client order ID
            
        Returns:
            Dict: Order information
        """
        try:
            order = self.client.get_order(
                symbol=symbol,
                orderId=order_id,
                origClientOrderId=orig_client_order_id
            )
            return order
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> Dict:
        """
        Cancel an order
        
        Args:
            symbol (str): Trading pair symbol
            order_id (int): Order ID
            orig_client_order_id (str): Original client order ID
            
        Returns:
            Dict: Cancellation confirmation
        """
        try:
            result = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id,
                origClientOrderId=orig_client_order_id
            )
            logger.info(f"Cancelled order {order_id} for {symbol}")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'error': str(e)}
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open orders
        
        Args:
            symbol (str): Trading pair symbol (optional)
            
        Returns:
            List[Dict]: Open orders
        """
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            return orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return [{'error': str(e)}]
    
    def get_all_orders(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get all orders for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            limit (int): Maximum number of orders to return
            
        Returns:
            List[Dict]: Orders
        """
        try:
            orders = self.client.get_all_orders(symbol=symbol, limit=limit)
            return orders
        except Exception as e:
            logger.error(f"Error getting all orders: {e}")
            return [{'error': str(e)}]
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a trading symbol
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Dict: Symbol information
        """
        try:
            exchange_info = self.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if symbol_info:
                return symbol_info
            else:
                logger.warning(f"Symbol {symbol} not found")
                return {'error': f"Symbol {symbol} not found"}
                
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_websocket_depth_stream(self, symbol: str) -> str:
        """
        Get the websocket stream URL for order book depth
        
        Args:
            symbol (str): Trading pair symbol (lowercase)
            
        Returns:
            str: Websocket stream URL
        """
        symbol = symbol.lower()
        return f"wss://stream.binance.com:9443/ws/{symbol}@depth"
    
    def get_websocket_kline_stream(self, symbol: str, interval: str) -> str:
        """
        Get the websocket stream URL for klines
        
        Args:
            symbol (str): Trading pair symbol (lowercase)
            interval (str): Kline interval
            
        Returns:
            str: Websocket stream URL
        """
        symbol = symbol.lower()
        return f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    
    def get_websocket_ticker_stream(self, symbol: str) -> str:
        """
        Get the websocket stream URL for ticker updates
        
        Args:
            symbol (str): Trading pair symbol (lowercase)
            
        Returns:
            str: Websocket stream URL
        """
        symbol = symbol.lower()
        return f"wss://stream.binance.com:9443/ws/{symbol}@ticker"