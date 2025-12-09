from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import time
from datetime import datetime, timedelta

router = APIRouter(
    prefix="/api/v1/analysis",
    tags=["analysis"]
)

class SentimentData(BaseModel):
    score: float  # -1 to 1
    label: str    # Bullish, Bearish, Neutral
    fear_greed_index: int # 0-100
    sources: Dict[str, float] # e.g., {"twitter": 0.6, "reddit": 0.4}
    timestamp: float

class PredictionData(BaseModel):
    symbol: str
    direction: str # UP, DOWN, SIDEWAYS
    confidence: float # 0-1
    target_price: float
    timeframe: str
    factors: List[str]
    timestamp: float

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    timeframe: str
    start_date: str
    end_date: str
    parameters: Dict[str, float]

class BacktestResult(BaseModel):
    total_return: float
    max_drawdown: float
    win_rate: float
    trades_count: int
    equity_curve: List[Dict[str, float]] # timestamp, value

class NewsItem(BaseModel):
    id: str
    title: str
    source: str
    url: str
    published_at: str
    sentiment: str # Positive, Negative, Neutral

@router.get("/sentiment/{symbol}", response_model=SentimentData)
def get_sentiment(symbol: str):
    """Get aggregated sentiment data for a symbol (Mocked)"""
    # Mock logic based on random seed for consistency
    random.seed(symbol + str(int(time.time() / 3600))) # Changes every hour
    
    score = random.uniform(-0.8, 0.8)
    fear_greed = random.randint(20, 80)
    
    label = "Neutral"
    if score > 0.3: label = "Bullish"
    if score < -0.3: label = "Bearish"
    
    return {
        "score": round(score, 2),
        "label": label,
        "fear_greed_index": fear_greed,
        "sources": {
            "twitter": round(random.uniform(-1, 1), 2),
            "reddit": round(random.uniform(-1, 1), 2),
            "news": round(random.uniform(-1, 1), 2)
        },
        "timestamp": time.time()
    }

@router.get("/prediction/{symbol}", response_model=PredictionData)
def get_prediction(symbol: str):
    """Get AI-driven price prediction (Mocked)"""
    # Mock logic
    random.seed(symbol + str(int(time.time() / 300))) # Changes every 5 mins
    
    current_price = 50000 # Base price for mock
    direction = random.choice(["UP", "DOWN", "SIDEWAYS"])
    change = random.uniform(0.01, 0.05)
    
    target = current_price * (1 + change) if direction == "UP" else current_price * (1 - change)
    if direction == "SIDEWAYS": target = current_price
    
    factors = [
        "RSI Divergence" if random.random() > 0.5 else "MACD Crossover",
        "High Volume" if random.random() > 0.5 else "Support Level Test"
    ]
    
    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": round(random.uniform(0.6, 0.95), 2),
        "target_price": round(target, 2),
        "timeframe": "24h",
        "factors": factors,
        "timestamp": time.time()
    }

@router.post("/backtest", response_model=BacktestResult)
def run_backtest(request: BacktestRequest):
    """Run a strategy backtest (Mocked simulation)"""
    # Simulate processing time
    time.sleep(1)
    
    # Generate a random equity curve
    equity = 10000
    curve = []
    now = datetime.now()
    
    for i in range(30):
        date = (now - timedelta(days=30-i)).isoformat()
        change = random.uniform(-0.02, 0.03)
        equity *= (1 + change)
        curve.append({"time": date, "value": round(equity, 2)})
        
    total_return = (equity - 10000) / 10000 * 100
    
    return {
        "total_return": round(total_return, 2),
        "max_drawdown": round(random.uniform(5, 20), 2),
        "win_rate": round(random.uniform(40, 70), 2),
        "trades_count": random.randint(10, 50),
        "equity_curve": curve
    }

@router.get("/news", response_model=List[NewsItem])
def get_news(category: str = "all"):
    """Get crypto news feed (Mocked)"""
    headlines = [
        ("Bitcoin Breaks $100k Resistance", "CoinDesk", "Positive"),
        ("SEC Approves New Crypto ETF", "Bloomberg", "Positive"),
        ("Market Volatility Increases Ahead of Fed Meeting", "Reuters", "Neutral"),
        ("Major Exchange Hack Reported", "TechCrunch", "Negative"),
        ("Ethereum 2.0 Upgrade Successful", "Decrypt", "Positive"),
        ("Regulatory Uncertainty Continues", "CNBC", "Negative")
    ]
    
    news = []
    for i, (title, source, sentiment) in enumerate(headlines):
        news.append({
            "id": str(i),
            "title": title,
            "source": source,
            "url": "#",
            "published_at": (datetime.now() - timedelta(hours=i)).isoformat(),
            "sentiment": sentiment
        })
        
    return news
