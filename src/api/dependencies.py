from typing import Optional
from src.trading.trading_system import TradingSystem

# Global trading system instance
_trading_system: Optional[TradingSystem] = None

def set_trading_system(ts: TradingSystem):
    """Set the global trading system instance"""
    global _trading_system
    _trading_system = ts

def get_trading_system() -> TradingSystem:
    """
    Dependency injection for TradingSystem.
    This will be used in FastAPI route dependencies.
    """
    if _trading_system is None:
        raise RuntimeError("Trading system not initialized")
    return _trading_system
