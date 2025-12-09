import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CryptoVisionAI"
    VERSION: str = "1.0.0"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]
    
    # Binance API
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    BINANCE_USE_TESTNET: bool = True
    
    # Trading Settings
    TRADING_INITIAL_CAPITAL: float = 10000.0
    TRADING_MAX_POSITION_SIZE: float = 20.0
    TRADING_MAX_OPEN_POSITIONS: int = 5
    TRADING_STOP_LOSS_PERCENT: float = 3.0
    TRADING_TAKE_PROFIT_PERCENT: float = 6.0
    
    # Model Settings
    MODEL_PATH: str = "models/ensemble"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    DEBUG_MODE: bool = False
    USE_TESTNET: bool = True
    
    class Config:
        env_file = None if os.getenv("SKIP_DOTENV") else ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()
