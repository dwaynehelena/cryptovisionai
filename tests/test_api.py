import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import pandas as pd
from src.api.main import app
from src.api.dependencies import get_trading_system

client = TestClient(app)

# Mock TradingSystem
mock_trading_system = MagicMock()
app.dependency_overrides[get_trading_system] = lambda: mock_trading_system

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}

def test_get_market_data():
    # Mock return value
    mock_df = pd.DataFrame({
        'open': [100.0], 'high': [110.0], 'low': [90.0], 'close': [105.0], 'volume': [1000.0]
    }, index=pd.to_datetime(['2023-01-01']))
    mock_trading_system.binance_connector.get_historical_klines.return_value = mock_df

    response = client.get("/api/v1/market/data?symbol=BTCUSDT")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]['close'] == 105.0
    assert 'timestamp' in data[0]

def test_get_price():
    mock_trading_system.binance_connector.get_ticker.return_value = {'symbol': 'BTCUSDT', 'price': '30000.00'}
    
    response = client.get("/api/v1/market/price/BTCUSDT")
    assert response.status_code == 200
    assert response.json() == {'symbol': 'BTCUSDT', 'price': '30000.00'}

def test_get_portfolio():
    mock_summary = {
        "initial_capital": 10000.0,
        "current_capital": 10500.0,
        "portfolio_value": 10500.0,
        "profit_loss": 500.0,
        "profit_loss_percent": 5.0
    }
    mock_trading_system.portfolio.get_portfolio_summary.return_value = mock_summary

    response = client.get("/api/v1/portfolio/")
    assert response.status_code == 200
    assert response.json() == mock_summary

def test_get_positions():
    mock_positions = {
        "open_positions": [],
        "closed_positions": []
    }
    mock_trading_system.portfolio.get_all_positions.return_value = mock_positions

    response = client.get("/api/v1/portfolio/positions")
    assert response.status_code == 200
    assert response.json() == mock_positions

