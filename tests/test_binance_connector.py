import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing.binance_connector import BinanceConnector

class TestBinanceConnector(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        # Patch the Client class where it is used in the binance_connector module
        with patch('data_processing.binance_connector.Client', return_value=self.mock_client):
            self.connector = BinanceConnector(use_testnet=True, api_key='test', api_secret='test')

    def test_get_exchange_info(self):
        self.mock_client.get_exchange_info.return_value = {
            'symbols': [
                {'symbol': 'BTCUSDT', 'status': 'TRADING'},
                {'symbol': 'ETHUSDT', 'status': 'TRADING'},
                {'symbol': 'XRPUSDT', 'status': 'BREAK'}
            ]
        }
        symbols = self.connector.get_active_symbols(quote_asset='USDT')
        self.assertIn('BTCUSDT', symbols)
        self.assertIn('ETHUSDT', symbols)
        self.assertNotIn('XRPUSDT', symbols)

    def test_get_historical_klines(self):
        # Mock kline data: [timestamp, open, high, low, close, volume, ...]
        mock_klines = [
            [1600000000000, '10000.0', '10100.0', '9900.0', '10050.0', '10.0', 1600000060000, '100000.0', 100, '5.0', '50000.0', '0'],
            [1600000060000, '10050.0', '10200.0', '10000.0', '10150.0', '15.0', 1600000120000, '150000.0', 150, '7.0', '70000.0', '0']
        ]
        self.mock_client.get_klines.return_value = mock_klines
        
        df = self.connector.get_historical_klines('BTCUSDT', '1m', limit=2)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('close', df.columns)
        self.assertIn('SMA_7', df.columns) # Check if technical indicators are added
        self.assertEqual(df.index[0], pd.to_datetime(1600000000000, unit='ms'))

    def test_create_order(self):
        self.mock_client.create_order.return_value = {'orderId': 12345, 'symbol': 'BTCUSDT'}
        result = self.connector.create_order('BTCUSDT', 'BUY', 'LIMIT', quantity=0.1, price=50000)
        self.assertEqual(result['orderId'], 12345)
        self.mock_client.create_order.assert_called_with(
            symbol='BTCUSDT', side='BUY', type='LIMIT', quantity=0.1, quoteOrderQty=None, 
            price=50000, timeInForce='GTC', stopPrice=None
        )

if __name__ == '__main__':
    unittest.main()
