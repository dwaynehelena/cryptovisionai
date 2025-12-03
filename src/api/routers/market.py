from fastapi import APIRouter, HTTPException, Depends
from src.trading.trading_system import TradingSystem
from src.api.dependencies import get_trading_system
from typing import List

router = APIRouter(prefix="/api/v1/market", tags=["market"])

@router.get("/data")
def get_market_data(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Get historical kline/candlestick data"""
    try:
        df = trading_system.binance_connector.get_historical_klines(symbol, interval, limit)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        # Convert timestamps to ISO format
        for record in data:
            if 'timestamp' in record:
                record['timestamp'] = record['timestamp'].isoformat()
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/price/{symbol}")
def get_market_price(symbol: str, trading_system: TradingSystem = Depends(get_trading_system)):
    """Get current ticker price for a symbol"""
    try:
        ticker = trading_system.binance_connector.get_ticker(symbol)
        
        if 'error' in ticker:
            raise HTTPException(status_code=404, detail=ticker['error'])
        
        return ticker
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/symbols")
def get_available_symbols(trading_system: TradingSystem = Depends(get_trading_system)):
    """Get list of available trading symbols"""
    try:
        # Get exchange info from Binance
        exchange_info = trading_system.binance_connector.client.get_exchange_info()
        
        # Extract trading pairs
        symbols = []
        for symbol_info in exchange_info['symbols']:
            if symbol_info['status'] == 'TRADING':
                symbols.append({
                    'symbol': symbol_info['symbol'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                    'status': symbol_info['status']
                })
        
        return symbols
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orderbook/{symbol}")
def get_order_book(symbol: str, limit: int = 20, trading_system: TradingSystem = Depends(get_trading_system)):
    """Get order book depth for a symbol"""
    try:
        order_book = trading_system.binance_connector.client.get_order_book(symbol=symbol, limit=limit)
        
        return {
            'symbol': symbol,
            'bids': order_book['bids'],
            'asks': order_book['asks'],
            'lastUpdateId': order_book['lastUpdateId']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trades/{symbol}")
def get_recent_trades(symbol: str, limit: int = 50, trading_system: TradingSystem = Depends(get_trading_system)):
    """Get recent trades for a symbol"""
    try:
        trades = trading_system.binance_connector.client.get_recent_trades(symbol=symbol, limit=limit)
        
        return [{
            'id': trade['id'],
            'price': trade['price'],
            'qty': trade['qty'],
            'time': trade['time'],
            'isBuyerMaker': trade['isBuyerMaker']
        } for trade in trades]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/indicators/{symbol}")
def get_technical_indicators(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Get technical indicators for a symbol"""
    try:
        df = trading_system.binance_connector.get_historical_klines(symbol, interval, limit)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = [high_low, high_close, low_close]
        # We need pandas to concat, but if it's not imported we can't use pd.concat.
        # Alternatively, we can use max of the three series.
        # df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # If I can't import pandas easily without checking if it's installed (it is), I should add the import.
        
        # Let's try to add the import at the top first in a separate call or just use a workaround if possible?
        # No, I should add the import.

        
        # Convert to dict
        data = df.to_dict('records')
        
        # Convert timestamps to ISO format
        for record in data:
            if 'timestamp' in record:
                record['timestamp'] = record['timestamp'].isoformat()
        
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
