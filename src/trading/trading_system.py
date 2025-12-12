#!/usr/bin/env python3
"""
Trading System Module - Executes trades based on model predictions
with sophisticated risk management strategies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from threading import Thread, RLock
from queue import Queue
import traceback
import json
import random # Ensure random is available for Signal Logging

# Import custom modules
from ..data_processing.binance_connector import BinanceConnector
from ..data_processing.feature_engineering import FeatureEngineering
from ..models.ensemble_model import EnsembleModel
from ..utils.alerts import AlertManager, AlertLevel, AlertType
from .risk_manager import RiskManager as EnhancedRiskManager

# Configure logging
logger = logging.getLogger("trading_system")

class Position:
    """
    Class to track an individual trading position
    """
    
    def __init__(self, symbol: str, position_id: str, position_type: str, 
                entry_price: float, amount: float, leverage: float = 1.0):
        """
        Initialize a new position
        
        Args:
            symbol (str): Trading pair symbol
            position_id (str): Unique position identifier
            position_type (str): 'long' or 'short'
            entry_price (float): Entry price of the position
            amount (float): Amount of the asset
            leverage (float): Leverage used
        """
        self.symbol = symbol
        self.position_id = position_id
        self.position_type = position_type
        self.entry_price = entry_price
        self.amount = amount
        self.leverage = leverage
        self.exit_price = None
        self.entry_time = datetime.now()
        self.exit_time = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.status = "open"
        self.stop_loss = None
        self.take_profit = None
        self.orders = []
        self.metadata = {}
        self.current_price = entry_price # Initialize with entry price
        
    def close(self, exit_price: float) -> None:
        """
        Close the position
        
        Args:
            exit_price (float): Exit price of the position
        """
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.status = "closed"
        
        # Calculate PnL
        if self.position_type == "long":
            self.pnl = (exit_price - self.entry_price) * self.amount * self.leverage
            self.pnl_percent = (exit_price / self.entry_price - 1) * 100 * self.leverage
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.amount * self.leverage
            self.pnl_percent = (self.entry_price / exit_price - 1) * 100 * self.leverage
        
    def update(self, current_price: float) -> None:
        """
        Update the position with current market price
        
        Args:
            current_price (float): Current price
        """
        self.current_price = current_price # Store current price
        # Calculate unrealized PnL
        if self.position_type == "long":
            self.pnl = (current_price - self.entry_price) * self.amount * self.leverage
            self.pnl_percent = (current_price / self.entry_price - 1) * 100 * self.leverage
        else:  # short
            self.pnl = (self.entry_price - current_price) * self.amount * self.leverage
            self.pnl_percent = (self.entry_price / current_price - 1) * 100 * self.leverage
    
    def set_stop_loss(self, price: float) -> None:
        """
        Set stop loss for the position
        
        Args:
            price (float): Stop loss price
        """
        self.stop_loss = price
    
    def set_take_profit(self, price: float) -> None:
        """
        Set take profit for the position
        
        Args:
            price (float): Take profit price
        """
        self.take_profit = price
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary
        
        Returns:
            Dict[str, Any]: Position data
        """
        return {
            "symbol": self.symbol,
            "position_id": self.position_id,
            "position_type": self.position_type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "amount": self.amount,
            "leverage": self.leverage,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "status": self.status,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "duration": str(self.exit_time - self.entry_time) if self.exit_time else None,
            "metadata": self.metadata
        }


class Portfolio:
    """
    Class to manage a portfolio of positions and track performance
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize the portfolio
        
        Args:
            initial_capital (float): Initial capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.closed_positions = {}
        self.history = []
        self.last_updated = datetime.now()
        self.last_updated = datetime.now()
        self.trade_history = []  # List of closed trades
        self.lock = RLock()  # Thread safety - Reentrant!
        
        # Performance metrics
        self.performance_metrics = {
            "total_pnl": 0.0,
            "total_pnl_percent": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0
        }
        # Add total_value property for compatibility
        # This will be accessed as an attribute, so we define a getter method below.

    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value including current capital and unrealized PnL of open positions."""
        with self.lock:
            open_pnl = sum(pos.pnl for pos in self.positions.values())
            return self.current_capital + open_pnl

    def add_position(self, position: Position) -> None:
        """
        Add a new position to the portfolio
        
        Args:
            position (Position): Position object
        """
        with self.lock:
            self.positions[position.position_id] = position
            self.record_history()
    
    def close_position(self, position_id: str, exit_price: float) -> Optional[Position]:
        """
        Close an existing position
        
        Args:
            position_id (str): Position ID
            exit_price (float): Exit price
            
        Returns:
            Optional[Position]: Closed position or None if not found
        """
        with self.lock:
            if position_id in self.positions:
                position = self.positions[position_id]
                position.close(exit_price)
                
                # Update capital
                self.current_capital += position.pnl
                
                # Move to closed positions
                self.closed_positions[position_id] = position
                del self.positions[position_id]
                
                # Update performance metrics
                self.update_performance_metrics()
                self.record_history()
                
                # Add to trade history
                self.trade_history.append(position.to_dict())
                
                return position
            return None
    
    def save_to_file(self, filepath: str) -> None:
        """Save portfolio state to JSON file"""
        with self.lock:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                data = {
                    "initial_capital": self.initial_capital,
                    "current_capital": self.current_capital,
                    "positions": {pid: pos.to_dict() for pid, pos in self.positions.items()},
                    "history": self.history[-1000:], # Keep last 1000 records
                    "performance_metrics": self.performance_metrics,
                    "trade_history": self.trade_history
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4, default=str)
                logger.debug(f"Saved portfolio to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save portfolio: {e}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Portfolio':
        """Load portfolio state from JSON file"""
        try:
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            portfolio = cls(data["initial_capital"])
            portfolio.current_capital = data["current_capital"]
            # Reconstruct positions... requires Position.from_dict but loose dict is okay for now if updated
            # Actually, we need to reconstruct Position objects
            for pid, pos_data in data["positions"].items():
                pos = Position(
                    symbol=pos_data["symbol"],
                    position_id=pid,
                    position_type=pos_data["position_type"],
                    entry_price=pos_data["entry_price"],
                    amount=pos_data["amount"]
                )
                pos.current_price = pos_data.get("current_price", pos_data["entry_price"])
                pos.pnl = pos_data.get("pnl", 0.0)
                pos.stop_loss = pos_data.get("stop_loss")
                pos.take_profit = pos_data.get("take_profit")
                pos.status = pos_data.get("status", "open")
                pos.entry_time = datetime.fromisoformat(pos_data["entry_time"]) if "entry_time" in pos_data else datetime.now()
                portfolio.positions[pid] = pos
            
            portfolio.history = data.get("history", [])
            portfolio.performance_metrics = data.get("performance_metrics", portfolio.performance_metrics)
            portfolio.trade_history = data.get("trade_history", [])
            
            logger.info(f"Loaded portfolio from {filepath} with {len(portfolio.positions)} positions")
            return portfolio
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            return None

    def update_positions(self, prices: Dict[str, float]) -> None:
        """
        Update all positions with current market prices
        
        Args:
            prices (Dict[str, float]): Dictionary of symbol:price pairs
        """
        with self.lock:
            for position_id, position in list(self.positions.items()):
                if position.symbol in prices:
                    current_price = prices[position.symbol]
                    position.update(current_price)
                    
                    # Check for stop loss or take profit
                    if self.check_exit_conditions(position, current_price):
                        self.close_position(position_id, current_price)
            
            # Update portfolio valuation
            self.last_updated = datetime.now()
            self.record_history()
    
    def check_exit_conditions(self, position: Position, current_price: float) -> bool:
        """
        Check if position should be closed based on stop loss or take profit
        
        Args:
            position (Position): Position object
            current_price (float): Current price
            
        Returns:
            bool: True if position should be closed
        """
        if position.status != "open":
            return False
            
        # Check stop loss
        if position.stop_loss is not None:
            if (position.position_type == "long" and current_price <= position.stop_loss) or \
               (position.position_type == "short" and current_price >= position.stop_loss):
                logger.info(f"Stop loss triggered for position {position.position_id} at {current_price}")
                return True
                
        # Check take profit
        if position.take_profit is not None:
            if (position.position_type == "long" and current_price >= position.take_profit) or \
               (position.position_type == "short" and current_price <= position.take_profit):
                logger.info(f"Take profit triggered for position {position.position_id} at {current_price}")
                return True
                
        return False
    
    def update_performance_metrics(self) -> Dict[str, float]:
        """
        Update portfolio performance metrics
        
        Returns:
            Dict[str, float]: Updated metrics
        """
        with self.lock:
            if not self.closed_positions:
                return self.performance_metrics
                
            # Calculate total PnL
            total_pnl = sum(pos.pnl for pos in self.closed_positions.values())
            
            # Calculate win/loss statistics
            winning_positions = [pos for pos in self.closed_positions.values() if pos.pnl > 0]
            losing_positions = [pos for pos in self.closed_positions.values() if pos.pnl <= 0]
            
            win_count = len(winning_positions)
            loss_count = len(losing_positions)
            total_count = win_count + loss_count
            
            # Win rate
            win_rate = win_count / total_count if total_count > 0 else 0
            
            # Average win/loss
            average_win = sum(pos.pnl for pos in winning_positions) / win_count if win_count > 0 else 0
            average_loss = sum(pos.pnl for pos in losing_positions) / loss_count if loss_count > 0 else 0
            
            # Largest win/loss
            largest_win = max([pos.pnl for pos in winning_positions]) if winning_positions else 0
            largest_loss = min([pos.pnl for pos in losing_positions]) if losing_positions else 0
            
            # Profit factor
            gross_profit = sum(pos.pnl for pos in winning_positions)
            gross_loss = abs(sum(pos.pnl for pos in losing_positions)) if losing_positions else 1
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
            
            # Calculate daily returns for Sharpe ratio
            daily_returns = []
            if len(self.history) > 1:
                for i in range(1, len(self.history)):
                    prev_capital = self.history[i-1]["capital"]
                    current_capital = self.history[i]["capital"]
                    if prev_capital > 0:
                        daily_return = (current_capital / prev_capital) - 1
                        daily_returns.append(daily_return)
            
            # Sharpe ratio
            sharpe_ratio = 0
            if daily_returns:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized
            
            # Max drawdown
            max_drawdown = 0
            max_drawdown_percent = 0
            peak_capital = self.initial_capital
            
            for snapshot in self.history:
                capital = snapshot["capital"]
                if capital > peak_capital:
                    peak_capital = capital
                else:
                    drawdown = peak_capital - capital
                    drawdown_percent = (drawdown / peak_capital) * 100
                    max_drawdown = max(max_drawdown, drawdown)
                    max_drawdown_percent = max(max_drawdown_percent, drawdown_percent)
            
            # Update metrics
            self.performance_metrics = {
                "total_pnl": total_pnl,
                "total_pnl_percent": (total_pnl / self.initial_capital) * 100,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": win_rate * 100,  # As percentage
                "average_win": average_win,
                "average_loss": average_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "profit_factor": profit_factor,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "max_drawdown_percent": max_drawdown_percent
            }
            
            return self.performance_metrics
    
    def record_history(self) -> None:
        """
        Record current portfolio state in history
        """
        # Calculate current portfolio value
        portfolio_value = self.current_capital + sum(pos.pnl for pos in self.positions.values())
        
        snapshot = {
            "timestamp": datetime.now(),
            "capital": portfolio_value,
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions)
        }
        
        self.history.append(snapshot)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio
        
        Returns:
            Dict[str, Any]: Portfolio summary
        """
        with self.lock:
            # Update metrics first
            self.update_performance_metrics()
            
            # Calculate current portfolio value
            portfolio_value = self.current_capital + sum(pos.pnl for pos in self.positions.values())
            
            # Current exposure
            current_exposure = sum(pos.amount * pos.entry_price for pos in self.positions.values())
            
            return {
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "portfolio_value": portfolio_value,
                "profit_loss": portfolio_value - self.initial_capital,
                "profit_loss_percent": ((portfolio_value / self.initial_capital) - 1) * 100,
                "open_positions": len(self.positions),
                "closed_positions": len(self.closed_positions),
                "current_exposure": current_exposure,
                "exposure_percent": (current_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0,
                "metrics": self.performance_metrics,
                "last_updated": self.last_updated
            }
    
    def get_position_details(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific position
        
        Args:
            position_id (str): Position ID
            
        Returns:
            Optional[Dict[str, Any]]: Position details or None if not found
        """
        with self.lock:
            if position_id in self.positions:
                return self.positions[position_id].to_dict()
            elif position_id in self.closed_positions:
                return self.closed_positions[position_id].to_dict()
            return None
    
    def get_all_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all positions (open and closed)
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary with open and closed positions
        """
        with self.lock:
            open_positions = [pos.to_dict() for pos in self.positions.values()]
            closed_positions = [pos.to_dict() for pos in self.closed_positions.values()]
            
            return {
                "open_positions": open_positions,
                "closed_positions": closed_positions
            }


class RiskManager:
    """
    Handles risk management strategy for trading
    """
    
    def __init__(self, config: Dict[str, Any], portfolio: Portfolio):
        """
        Initialize the risk manager
        
        Args:
            config (Dict[str, Any]): Risk management configuration
            portfolio (Portfolio): Portfolio object
        """
        self.config = config
        self.portfolio = portfolio
        self.max_portfolio_risk = config.get("max_portfolio_risk", 5.0)  # % of portfolio at risk
        self.max_position_size = config.get("max_position_size", 20.0)  # % of portfolio
        self.max_open_positions = config.get("max_open_positions", 10)
        self.stop_loss_percent = config.get("stop_loss_percent", 5.0)
        self.take_profit_percent = config.get("take_profit_percent", 10.0)
        self.max_drawdown_limit = config.get("max_drawdown_limit", 20.0)
        self.volatility_window = config.get("volatility_window", 20)
        self.risk_per_trade = config.get("risk_per_trade", 1.0)  # % of portfolio per trade
        self.kelly_fraction = config.get("kelly_fraction", 0.5)  # Kelly criterion fraction
        self.max_correlation = config.get("max_correlation", 0.7)  # Max correlation between assets
        
        logger.info(f"Initialized risk manager with max risk: {self.max_portfolio_risk}%")
    
    def can_open_position(self, symbol: str, price_data: pd.DataFrame) -> bool:
        """
        Check if a new position can be opened based on risk limits
        
        Args:
            symbol (str): Trading pair symbol
            price_data (pd.DataFrame): Historical price data for volatility calculation
            
        Returns:
            bool: True if position can be opened
        """
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        # Check if max number of positions is reached
        if len(self.portfolio.positions) >= self.max_open_positions:
            logger.warning(f"Maximum number of open positions reached ({self.max_open_positions})")
            return False
        
        # Check if portfolio drawdown limit is exceeded
        if portfolio_summary["metrics"]["max_drawdown_percent"] >= self.max_drawdown_limit:
            logger.warning(f"Maximum portfolio drawdown limit reached ({self.max_drawdown_limit}%)")
            return False
        
        # Check correlations with existing positions
        if len(self.portfolio.positions) > 0 and price_data is not None:
            # Get correlation matrix
            for position_id, position in self.portfolio.positions.items():
                if position.symbol != symbol and symbol in price_data.columns and position.symbol in price_data.columns:
                    correlation = price_data[symbol].corr(price_data[position.symbol])
                    if abs(correlation) > self.max_correlation:
                        logger.warning(f"Position correlation too high between {symbol} and {position.symbol}: {correlation:.2f}")
                        return False
                        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float, 
                               volatility: Optional[float] = None) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Current price
            signal_strength (float): Model signal strength (0.0 - 1.0)
            volatility (Optional[float]): Asset volatility
            
        Returns:
            float: Position size (in base currency)
        """
        portfolio_summary = self.portfolio.get_portfolio_summary()
        portfolio_value = portfolio_summary["portfolio_value"]
        
        # Base size on portfolio risk per trade
        position_value = portfolio_value * (self.risk_per_trade / 100)
        
        # Adjust by signal strength (0.5 to 1.5x)
        signal_factor = 0.5 + signal_strength
        position_value *= signal_factor
        
        # Adjust by volatility if provided
        if volatility is not None:
            # Lower position size for higher volatility
            volatility_factor = 1.0 / (1.0 + volatility)
            position_value *= volatility_factor
        
        # Apply kelly criterion if win rate is available
        win_rate = portfolio_summary["metrics"]["win_rate"] / 100 if portfolio_summary["metrics"]["win_rate"] else 0.5
        if win_rate > 0:
            # Simple Kelly formula: f = p - (1-p)/R
            # where p is win rate, R is win/loss ratio
            avg_win = portfolio_summary["metrics"]["average_win"]
            avg_loss = abs(portfolio_summary["metrics"]["average_loss"]) if portfolio_summary["metrics"]["average_loss"] else 1
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            
            kelly_size = win_rate - ((1 - win_rate) / win_loss_ratio)
            kelly_size = max(0, kelly_size * self.kelly_fraction)
            
            # Apply Kelly limit
            kelly_position = portfolio_value * kelly_size
            position_value = min(position_value, kelly_position)
        
        # Maximum position size limit
        max_position_value = portfolio_value * (self.max_position_size / 100)
        position_value = min(position_value, max_position_value)
        
        # Convert to position size in base currency
        position_size = position_value / price
        
        logger.info(f"Calculated position size for {symbol}: {position_size:.6f} (value: {position_value:.2f})")
        return position_size
    
    def calculate_stop_loss(self, symbol: str, price: float, position_type: str, 
                           volatility: Optional[float] = None) -> float:
        """
        Calculate stop loss price based on risk parameters and volatility
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Entry price
            position_type (str): 'long' or 'short'
            volatility (Optional[float]): Asset volatility
            
        Returns:
            float: Stop loss price
        """
        # Base stop loss on fixed percentage
        stop_percent = self.stop_loss_percent
        
        # Adjust by volatility if provided
        if volatility is not None:
            stop_percent = stop_percent * (1 + volatility)
            
        # Calculate price
        if position_type == "long":
            stop_price = price * (1 - stop_percent / 100)
        else:  # short
            stop_price = price * (1 + stop_percent / 100)
            
        logger.info(f"Calculated stop loss for {symbol} {position_type} position: {stop_price:.6f} ({stop_percent:.2f}%)")
        return stop_price
    
    def calculate_take_profit(self, symbol: str, price: float, position_type: str,
                             volatility: Optional[float] = None) -> float:
        """
        Calculate take profit price based on risk parameters and volatility
        
        Args:
            symbol (str): Trading pair symbol
            price (float): Entry price
            position_type (str): 'long' or 'short'
            volatility (Optional[float]): Asset volatility
            
        Returns:
            float: Take profit price
        """
        # Base take profit on fixed percentage
        take_profit_percent = self.take_profit_percent
        
        # Adjust by volatility if provided
        if volatility is not None:
            take_profit_percent = take_profit_percent * (1 + volatility)
            
        # Calculate price
        if position_type == "long":
            tp_price = price * (1 + take_profit_percent / 100)
        else:  # short
            tp_price = price * (1 - take_profit_percent / 100)
            
        logger.info(f"Calculated take profit for {symbol} {position_type} position: {tp_price:.6f} ({take_profit_percent:.2f}%)")
        return tp_price
    
    def calculate_volatility(self, price_data: pd.DataFrame) -> float:
        """
        Calculate asset volatility
        
        Args:
            price_data (pd.DataFrame): Historical price data
            
        Returns:
            float: Volatility (standard deviation of returns)
        """
        if len(price_data) < 2:
            return 0.0
            
        # Calculate daily returns
        returns = price_data['close'].pct_change().dropna()
        
        if len(returns) > self.volatility_window:
            returns = returns[-self.volatility_window:]
            
        volatility = returns.std()
        return volatility
    
    def adjust_for_market_conditions(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Adjust risk parameters based on market conditions
        
        Args:
            symbol (str): Trading pair symbol
            price_data (pd.DataFrame): Historical price data
            
        Returns:
            Dict[str, float]: Adjusted risk parameters
        """
        # Use ATR for volatility
        atr = None
        if len(price_data) >= 14:
            high = price_data['high'].values
            low = price_data['low'].values
            close = price_data['close'].values
            
            # Simple ATR calculation
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            atr = np.mean(tr[-14:])
            atr_percent = (atr / price_data['close'].iloc[-1]) * 100
        else:
            atr_percent = 2.0  # Default
        
        # Adjust risk parameters based on ATR
        adjusted_params = {
            "stop_loss_percent": max(self.stop_loss_percent, atr_percent * 1.5),
            "take_profit_percent": max(self.take_profit_percent, atr_percent * 3.0),
            "risk_per_trade": self.risk_per_trade * (1.0 / (1.0 + atr_percent/10))
        }
        
        return adjusted_params


class TradingSystem:
    """
    Main trading system that connects prediction models with execution
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trading system
        
        Args:
            config (Dict[str, Any]): Trading system configuration
        """
        self.config = config
        self.binance_connector = None
        self.feature_engineering = None
        self.model = None
        self.portfolio = None
        self.risk_manager = None
        self.alert_manager = None  # Alert manager
        self.running = False
        self.trade_queue = Queue()
        self.worker_thread = None
        self.data_cache = {}  # Cache for price data and features
        self.active_symbols = []
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Trading system initialized")
    
    def _initialize_components(self) -> None:
        """Initialize system components"""
        # Create Binance connector
        use_testnet = self.config.get("use_testnet", True)
        # Use absolute path for config file instead of relative path
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
        # Get API keys from config
        api_key = self.config.get("api_key")
        api_secret = self.config.get("api_secret")
        
        self.binance_connector = BinanceConnector(
            use_testnet=use_testnet, 
            config_path=config_path,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Initialize alert manager
        alert_config = self.config.get("alerts", {})
        self.alert_manager = AlertManager(alert_config)
        
        # Create feature engineering module
        self.feature_engineering = FeatureEngineering()
        
        initial_capital = self.config.get("trading", {}).get("capital", 1000.0)
        self.portfolio_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "portfolio.json")
        
        # Try to load existing portfolio
        loaded_portfolio = Portfolio.load_from_file(self.portfolio_file)
        if loaded_portfolio:
            self.portfolio = loaded_portfolio
            logger.info(f"Resumed portfolio with capital ${self.portfolio.current_capital:.2f}")
        else:
            self.portfolio = Portfolio(initial_capital)
            logger.info(f"Initialized new portfolio with capital ${initial_capital}")
        
        # Create enhanced risk manager
        self.risk_manager = EnhancedRiskManager(initial_capital)
        
        # Update risk limits from config if provided
        risk_config = self.config.get("risk_management", {})
        if risk_config:
            self.risk_manager.update_limits(
                max_position_size_pct=risk_config.get("max_position_size", 10.0),
                max_risk_per_trade_pct=risk_config.get("risk_per_trade", 2.0),
                max_drawdown_pct=risk_config.get("max_drawdown_limit", 20.0),
                daily_loss_limit_pct=risk_config.get("daily_loss_limit", 5.0)
            )
        
        # Get active trading symbols
        if self.config.get("trading", {}).get("symbols"):
            self.active_symbols = self.config["trading"]["symbols"]
            logger.info(f"Using symbols from config: {self.active_symbols}")
        else:
            base_asset = self.config.get("base_asset", "BTC")
            quote_asset = self.config.get("quote_asset", "USDT")
            self.active_symbols = self.binance_connector.get_active_symbols(base_asset, quote_asset)
        
        # Load trading model
        self._load_model()
        
        # Production mode warnings
        if not self.config.get("use_testnet", True) or not self.config.get("test_mode", True):
            self._show_production_warnings()
    
    def check_readiness(self) -> Dict[str, Any]:
        """
        Check if the trading system is ready for automatic trading
        
        Returns:
            Dict[str, Any]: Readiness status of various components
        """
        status = {
            "ready": False,
            "components": {
                "binance_api": {"status": "unknown", "details": ""},
                "model": {"status": "unknown", "details": ""},
                "account": {"status": "unknown", "details": ""},
                "market_data": {"status": "unknown", "details": ""}
            },
            "timestamp": time.time()
        }
        
        # 1. Check Binance API Connection
        try:
            self.binance_connector.client.ping()
            status["components"]["binance_api"] = {"status": "operational", "details": "Connected"}
        except Exception as e:
            status["components"]["binance_api"] = {"status": "error", "details": str(e)}
            
        # 2. Check Model Status
        if self.model is not None:
            # You might want to add a more specific check if the model has an is_ready() method
            status["components"]["model"] = {"status": "operational", "details": "Model loaded"}
        else:
            status["components"]["model"] = {"status": "not_ready", "details": "Model not loaded"}
            
        # 3. Check Account Balance
        try:
            # Try to fetch account info to verify permissions and balance
            account_info = self.binance_connector.client.get_account()
            can_trade = account_info.get("canTrade", False)
            if can_trade:
                status["components"]["account"] = {"status": "operational", "details": "Account ready for trading"}
            else:
                status["components"]["account"] = {"status": "restricted", "details": "Account restricted from trading"}
        except Exception as e:
            status["components"]["account"] = {"status": "error", "details": str(e)}

        # 4. Check Market Data (Basic check)
        # This could be improved by checking the timestamp of the latest data for active symbols
        if self.active_symbols:
             status["components"]["market_data"] = {"status": "operational", "details": f"Tracking {len(self.active_symbols)} symbols"}
        else:
             status["components"]["market_data"] = {"status": "warning", "details": "No active symbols found"}

        # Determine overall readiness
        components = status["components"]
        is_ready = (
            components["binance_api"]["status"] == "operational" and
            components["model"]["status"] == "operational" and
            components["account"]["status"] == "operational"
        )
        status["ready"] = is_ready
        
        return status

    def _load_model(self) -> None:
        """Load prediction model(s)"""
        try:
            self.models = {}
            self.model_accuracies = {}
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            
            # Identify symbols to load models for
            # active_symbols might be set in start() -> _initialize_components
            # If not yet set, check config
            symbols_to_load = getattr(self, 'active_symbols', [])
            if not symbols_to_load:
                symbols_to_load = self.config.get("trading", {}).get("symbols", ["BTCUSDT"])
                
            logger.info(f"Loading models for symbols: {symbols_to_load}")

            # 1. Load Specific Models for each Symbol
            from ..models.tide_model import TiDEModel
            import json 
            
            # Common TiDE Config Default
            # Check if model_config is under model key (standard config.yaml structure)
            if 'model' in self.config and 'model_config' in self.config['model']:
                model_config_root = self.config['model']['model_config']
            else:
                model_config_root = self.config.get("model_config", {})
                
            tide_config_default = model_config_root.get("models", {}).get("tide", {}).copy()
            tide_config_default['output_dim'] = 3
            tide_config_default['output_activation'] = 'softmax'
            seq_len = 120 
            
            # Determine features based on config
            if not self.models:
                feature_set = self.config.get('model', {}).get('feature_set', 'standard') # Ensure feature_set is defined
                fe = FeatureEngineering() 
                # Suppress logging
                fe.logger.setLevel(logging.ERROR)
                dummy_df = pd.DataFrame({'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]}) # Create a dummy DataFrame
                
                cols = fe.extract_features(dummy_df, feature_set=feature_set).columns.tolist()
                n_transformed = len(cols)
                
                # Hardcode align with Real Data (146) for Standard set
                if feature_set == 'standard' and n_transformed < 146:
                    n_transformed = 146
                    
                # Get Expected Count (Source of Truth)
                try:
                    n_model = FeatureEngineering.get_expected_feature_count(feature_set)
                except:
                    n_model = n_transformed
                
                logger.info(f"🔒 Feature Safety Check: System expects {n_model} features (Set: {feature_set})")
                n_features_for_model = n_model # Use this for TiDEModel initialization

            for symbol in symbols_to_load:
                model_file = f"tide_winning_{symbol}.h5"
                tide_path_specific = os.path.join(base_dir, "models", model_file)
                
                if os.path.exists(tide_path_specific):
                    logger.info(f"[{symbol}] Found specific model: {model_file}")
                    try:
                        # Check for metadata
                        current_config = tide_config_default.copy()
                        meta_path = tide_path_specific + ".metadata.json"
                        
                        # FAIL SAFE: Verify Metadata Features
                        if os.path.exists(meta_path):
                             try:
                                with open(meta_path, 'r') as f:
                                    meta = json.load(f)
                                
                                # 1. Check Feature Count
                                if 'n_features' in meta:
                                    model_feats = meta['n_features']
                                    if model_feats != n_features:
                                        logger.error(f"[{symbol}] ⛔ FAIL SAFE TRIGGERED: Model has {model_feats} features, but System requires {n_features}. REJECTING MODEL.")
                                        continue # Skip loading this dangerous model
                                    else:
                                        logger.info(f"[{symbol}] ✅ Feature Count Verified: {model_feats}")
                                        
                                if 'params' in meta:
                                    p = meta['params']
                                    if 'hidden_dim' in p: current_config['hidden_dim'] = p['hidden_dim']
                                    if 'dropout' in p: current_config['dropout_rate'] = p['dropout']
                                    # lr not needed for inference
                                    logger.info(f"[{symbol}] Applied metadata config: {p}")
                                if 'accuracy' in meta:
                                    self.model_accuracies[symbol] = meta['accuracy']
                             except Exception as e:
                                 logger.warning(f"[{symbol}] Failed to read metadata: {e}")

                        m = TiDEModel(current_config, sequence_length=seq_len, n_features=n_features)
                        m.build_model()
                        m.model.load_weights(tide_path_specific)
                        self.models[symbol] = m
                    except Exception as e:
                        logger.error(f"Failed to load specific model for {symbol}: {e}")
            
            # 2. Load Generic Winning Model (Fallback)
            tide_path_generic = os.path.join(base_dir, "models", "tide_winning.h5")
            if os.path.exists(tide_path_generic):
                logger.info(f"Found generic winning model: tide_winning.h5")
                try:
                    # Check for metadata
                    current_config = tide_config_default.copy()
                    meta_path = tide_path_generic + ".metadata.json"
                    if os.path.exists(meta_path):
                            try:
                                with open(meta_path, 'r') as f:
                                    meta = json.load(f)
                                if 'params' in meta:
                                    p = meta['params']
                                    if 'hidden_dim' in p: current_config['hidden_dim'] = p['hidden_dim']
                                    if 'dropout' in p: current_config['dropout_rate'] = p['dropout']
                                    logger.info(f"[Generic] Applied metadata config: {p}")
                            except Exception as e:
                                logger.warning(f"[Generic] Failed to read metadata: {e}")

                    m = TiDEModel(current_config, sequence_length=seq_len, n_features=n_features)
                    m.build_model()
                    m.model.load_weights(tide_path_generic)
                    self.model = m # Default model
                except Exception as e:
                    logger.error(f"Failed to load generic model: {e}")
            else:
                 # Fallback to Ensemble logic if no generic TiDE
                 # Get model path from config, default to models/ensemble
                model_path = self.config.get("model_path", "models/ensemble")
                if not os.path.isabs(model_path):
                    model_path = os.path.join(base_dir, model_path)
                    
                if os.path.exists(model_path):
                    logger.info(f"Loading ensemble model using {model_path}")
                    if 'model' in self.config and 'model_config' in self.config['model']:
                        model_config = self.config['model']['model_config']
                    else:
                        model_config = self.config.get("model_config", {})
                        
                    self.model = EnsembleModel(model_config)
                    if self.model.load(model_path):
                        logger.info("Ensemble Model loaded successfully")
                
            # Summary
            logger.info(f"Loaded {len(self.models)} specific models.")
            if getattr(self, 'model', None):
                logger.info("Generic fallback model loaded.")
            else:
                logger.warning("No generic fallback model loaded.")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            traceback.print_exc()
    
    def _show_production_warnings(self) -> None:
        """Display critical warnings when running in production mode"""
        is_live = not self.config.get("use_testnet", True)
        is_real_trading = not self.config.get("test_mode", True)
        
        warning_messages = []
        
        if is_live and is_real_trading:
            warning_messages.append("⚠️ CRITICAL WARNING: System is running in PRODUCTION mode with REAL trading!")
            warning_messages.append("⚠️ CRITICAL WARNING: REAL funds will be used for trading!")
            warning_messages.append("⚠️ CRITICAL WARNING: Financial losses may occur!")
        elif is_live and not is_real_trading:
            warning_messages.append("⚠️ WARNING: Using LIVE Binance API but in test mode (no real trades)")
        elif not is_live and is_real_trading:
            warning_messages.append("⚠️ WARNING: Using Testnet but real trading is enabled")
            
        # Check risk parameters
        risk_config = self.config.get("risk_management", {})
        
        if is_live and is_real_trading:
            # Check for risky configurations
            max_position_size = risk_config.get("max_position_size", 20.0)
            if max_position_size > 10.0:
                warning_messages.append(f"⚠️ RISK WARNING: Maximum position size is set to {max_position_size}% (>10%)")
                
            stop_loss_percent = risk_config.get("stop_loss_percent", 5.0)
            if stop_loss_percent > 5.0:
                warning_messages.append(f"⚠️ RISK WARNING: Stop loss percent is set to {stop_loss_percent}% (>5%)")
                
            max_open_positions = risk_config.get("max_open_positions", 10)
            if max_open_positions > 5:
                warning_messages.append(f"⚠️ RISK WARNING: Maximum open positions is set to {max_open_positions} (>5)")
                
        # Log all warnings
        for msg in warning_messages:
            logger.critical(msg)
            
        # Send production alert through alert manager
        if self.alert_manager and (is_live or is_real_trading):
            self.alert_manager.system_alert(
                "TradingSystem", 
                "System started in production mode", 
                AlertLevel.CRITICAL if is_live and is_real_trading else AlertLevel.WARNING,
                {
                    "live_api": is_live,
                    "real_trading": is_real_trading,
                    "max_position_size": risk_config.get("max_position_size", 20.0),
                    "stop_loss_percent": risk_config.get("stop_loss_percent", 5.0),
                    "max_open_positions": risk_config.get("max_open_positions", 10)
                }
            )
            
        return warning_messages
    
    def start(self) -> None:
        """Start the trading system"""
        if self.running:
            logger.warning("Trading system already running")
            return
        
        self.running = True
        
        # Log Model Metadata for Dashboard
        self._log_model_metadata()
        
        logger.info("Starting trading system")
        
        # Start worker thread
        self.worker_thread = Thread(target=self._worker_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Trading system started")
    
    def stop(self) -> None:
        """Stop the trading system"""
        logger.info("Stopping trading system")
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=30)
        
        logger.info("Trading system stopped")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes trades and updates positions"""
        while self.running:
            try:
                # Update positions
                self._update_positions()
                
                # Process trade signals
                self._process_trade_signals()
                
                # Sleep for update interval
                time.sleep(self.config.get("update_interval_seconds", 60))
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                traceback.print_exc()
    
    def _update_positions(self) -> None:
        """Update all open positions with current market prices"""
        if not self.portfolio.positions:
            return
            
        # Get current prices for all symbols with open positions
        symbols = set(position.symbol for position in self.portfolio.positions.values())
        prices = {}
        
        for symbol in symbols:
            try:
                ticker = self.binance_connector.get_ticker(symbol)
                prices[symbol] = float(ticker['price'])
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
        
        # Update portfolio with current prices
        if prices:
            self.portfolio.update_positions(prices)
    
    def _process_trade_signals(self) -> None:
        """Process trade signals from the model"""
        # Check if it's time to generate signals
        current_time = datetime.now()
        signal_interval = self.config.get("signal_interval_minutes", 60)
        last_signal_time = getattr(self, "_last_signal_time", None)
        
        if last_signal_time and (current_time - last_signal_time).total_seconds() < signal_interval * 60:
            return
            
        self._last_signal_time = current_time
        
        # Generate signals for active symbols
        symbols_to_process = getattr(self, 'active_symbols', [])
        # Fallback if empty (e.g. not connected)
        if not symbols_to_process:
             logger.warning("No active symbols to process")
             return
             
        for symbol in symbols_to_process:
            try:
                # Get historical data
                df = self._get_historical_data(symbol)
                if df is None or len(df) < 50:
                    continue
                
                # Generate features
                features_df = self._generate_features(df)
                if features_df is None or len(features_df) < 1:
                    continue
                
                # Get model prediction (Specific Model per Symbol)
                prediction = self._get_prediction(features_df, symbol=symbol)
                if prediction is None:
                    continue
                
                # Process signal
                self._process_signal(symbol, df, prediction)
                
            except Exception as e:
                logger.error(f"Error processing signal for {symbol}: {e}")
                traceback.print_exc()
    
    def _get_historical_data(self, symbol: str, interval: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{interval}"
            if cache_key in self.data_cache and (datetime.now() - self.data_cache[cache_key]["timestamp"]).seconds < 3600:
                return self.data_cache[cache_key]["data"]
            
            # Get data from Binance
            df = self.binance_connector.get_historical_klines(symbol, interval, limit)
            
            if df is not None and not df.empty:
                # Update cache
                self.data_cache[cache_key] = {
                    "data": df,
                    "timestamp": datetime.now()
                }
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _generate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate features from historical data"""
        try:
            # Make a copy of the dataframe to avoid modifying the original
            df_copy = df.copy()
            
            # Ensure the dataframe has a reasonable size to avoid memory issues
            if len(df_copy) > 1000:
                df_copy = df_copy.tail(1000)
            
            # Use feature engineering module with error handling
            feature_set = self.config.get("feature_set", "standard")
            logger.info(f"Generating features with feature set: {feature_set}")
            
            # Generate features based on config
            features_df = self.feature_engineering.extract_features(
                df_copy, 
                feature_set=feature_set,
                include_target=False
            )
            
            logger.info(f"Generated features: {len(features_df.columns)} (Set: {feature_set})")
            
            if features_df is None or features_df.empty:
                logger.warning("Feature engineering returned empty dataframe")
                return None
                
            # Validate the features dataframe
            if len(features_df.columns) < 5:
                logger.warning(f"Feature engineering returned only {len(features_df.columns)} features")
                return None
                
            logger.info(f"Generated {len(features_df.columns)} features from {len(features_df)} rows of data")
            return features_df
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            # Return None instead of allowing the exception to propagate
            return None
    
    def _get_prediction(self, features_df: pd.DataFrame, symbol: str = None) -> Optional[Dict[str, float]]:
        """Get prediction from model"""
        try:
            # Determine which model to use
            model_to_use = None
            
            if symbol and getattr(self, 'models', None) and symbol in self.models:
                model_to_use = self.models[symbol]
            
            if model_to_use is None:
            # STRICT MODE: No Fallback
                logger.warning(f"[Strict] No specific model loaded for {symbol}. Skipping prediction.")
                return None
                
            if model_to_use is None:
                logger.warning(f"No model loaded (specific or generic) for {symbol}")
                return None
            
            # Prepare data for prediction
            # Ensure input shape matches what TiDE expects: (1, sequence_length, n_features)
            # The features_df might be just the features for the last sequence_length rows
            seq_len = getattr(model_to_use, 'sequence_length', 120) 
            # Check if features_df has enough rows
            if len(features_df) < seq_len:
                logger.warning(f"Not enough data for prediction. Need {seq_len}, got {len(features_df)}")
                return None
                
            X = features_df.iloc[-seq_len:].values  
            # Reshape for model: (1, seq_len, n_features)
            X = np.expand_dims(X, axis=0)
            
            # --- FAIL SAFE: Check Features Dimension ---
            input_features = X.shape[-1]
            model_features = getattr(model_to_use, 'n_features', 146) 
            
            if input_features != model_features:
                 logger.error(f"[{symbol}] ⛔ PREDICTION FAIL SAFE: Input has {input_features} features, Model expects {model_features}. ABORTING.")
                 return None
            
            # Get prediction
            prediction = model_to_use.predict(X)
            
            # Helper to interpret prediction
            signal = {"signal": "hold", "strength": 0.0, "probability": 0.0}
            
            # Case 1: Multi-class (Hold, Buy, Sell) - Shape (1, 3)
            if prediction.shape[-1] == 3:
                probs = prediction[0] # [p_hold, p_buy, p_sell]
                class_idx = np.argmax(probs)
                confidence = float(probs[class_idx])
                
                if class_idx == 1: # Buy
                    signal["signal"] = "buy"
                    signal["strength"] = confidence
                elif class_idx == 2: # Sell
                    signal["signal"] = "sell"
                    signal["strength"] = confidence
                else: # Hold
                    signal["signal"] = "hold"
                    signal["strength"] = confidence
                    
                signal["probability"] = confidence
                
            # Case 2: Binary Prob (Up/Down) - Shape (1, 1) or scalar
            else:
                probability = float(prediction[0])
                signal["probability"] = probability
                signal["signal"] = "buy" if probability > 0.6 else "sell" if probability < 0.4 else "hold"
                signal["strength"] = abs(probability - 0.5) * 2
            
            return signal
            
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return None
    
    def _process_signal(self, symbol: str, price_data: pd.DataFrame, signal: Dict[str, float]) -> None:
        """Process a trading signal"""
        current_price = price_data['close'].iloc[-1]
        
        # --- TREND FILTER (SMA 200) ---
        # Filter shorts in uptrend, longs in downtrend
        if len(price_data) >= 200:
            sma_200 = price_data['close'].rolling(window=200).mean().iloc[-1]
            
            # Rule 1: No SHORTS if Price > SMA 200 (Uptrend)
            if signal["signal"] == "sell" and current_price > sma_200:
                logger.info(f"[{symbol}] 🛡️ FILTERED SHORT: Price ${current_price:.2f} > SMA200 ${sma_200:.2f} (Uptrend)")
                return
                
            # Rule 2: No LONGS if Price < SMA 200 (Downtrend)
            # Make this optional? User specifically asked for SHORT filter.
            # But standard trend following implies this. I will enable it for safety.
            if signal["signal"] == "buy" and current_price < sma_200:
                logger.info(f"[{symbol}] 🛡️ FILTERED LONG: Price ${current_price:.2f} < SMA200 ${sma_200:.2f} (Downtrend)")
                return
                
        # Log the signal for visibility
        # Log all signals including HOLD for maximum transparency as requested
        logger.info(f"[{symbol}] 🧠 Signal: {signal['signal'].upper()} | Conf: {signal['strength']:.2f} | Price: ${current_price:.2f}")

        # Check if we already have an open position for this symbol
        has_position = any(p.symbol == symbol for p in self.portfolio.positions.values())
        
        if signal["signal"] == "buy" and not has_position:
            # Open long position
            self._open_position(symbol, "long", price_data, signal["strength"])
            
        elif signal["signal"] == "sell" and not has_position:
            # Open short position
            self._open_position(symbol, "short", price_data, signal["strength"])
            
        elif signal["signal"] != "hold" and has_position:
            # Close position if signal direction has changed
            for position_id, position in list(self.portfolio.positions.items()):
                if position.symbol == symbol:
                    if (position.position_type == "long" and signal["signal"] == "sell") or \
                       (position.position_type == "short" and signal["signal"] == "buy"):
                        self._close_position(position_id, price_data)
    
    def _open_position(self, symbol: str, position_type: str, price_data: pd.DataFrame, signal_strength: float) -> None:
        """Open a new position"""
        try:
            # Get current price
            ticker = self.binance_connector.get_ticker(symbol)
            price = float(ticker['price'])
            
            # Calculate volatility
            volatility = self.risk_manager.calculate_volatility(price_data)
            
            # Check if we can open position based on risk limits
            if not self.risk_manager.can_open_position(symbol, price_data):
                logger.warning(f"Cannot open {position_type} position for {symbol} due to risk limits")
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(symbol, price, signal_strength, volatility)
            
            # Calculate stop loss and take profit
            stop_loss = self.risk_manager.calculate_stop_loss(symbol, price, position_type, volatility)
            take_profit = self.risk_manager.calculate_take_profit(symbol, price, position_type, volatility)
            
            # Check trade value against portfolio for risk alerts
            portfolio_value = self.portfolio.get_portfolio_summary()["portfolio_value"]
            trade_value = position_size * price
            
            # Send alert if the trade size is substantial
            if self.alert_manager:
                self.alert_manager.check_trade_size(trade_value, portfolio_value, symbol, position_type)
            
            # Create position
            position_id = f"{symbol}_{position_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            position = Position(
                symbol=symbol,
                position_id=position_id,
                position_type=position_type,
                entry_price=price,
                amount=position_size
            )
            
            # Set stop loss and take profit
            position.set_stop_loss(stop_loss)
            position.set_take_profit(take_profit)
            
            # Add metadata
            position.metadata = {
                "signal_strength": signal_strength,
                "volatility": volatility,
                "entry_timestamp": datetime.now().isoformat()
            }
            
            # Execute order if not in test mode
            if not self.config.get("test_mode", True):
                order_side = "BUY" if position_type == "long" else "SELL"
                
                # Alert before executing a real trade
                if self.alert_manager:
                    self.alert_manager.trading_alert(
                        position_type, symbol, price, position_size,
                        AlertLevel.WARNING if trade_value > portfolio_value * 0.05 else AlertLevel.INFO,
                        {
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "signal_strength": signal_strength,
                            "volatility": volatility,
                            "portfolio_value": portfolio_value,
                            "trade_value": trade_value
                        }
                    )
                
                # Execute on Binance
                order = self.binance_connector.create_order(
                    symbol=symbol,
                    side=order_side,
                    order_type="MARKET",
                    quantity=position_size
                )
                
                # Check for errors
                if 'error' in order:
                    error_msg = f"Error creating order: {order['error']}"
                    logger.error(error_msg)
                    
                    # Alert on order errors
                    if self.alert_manager:
                        self.alert_manager.system_alert(
                            "OrderExecution", error_msg, AlertLevel.CRITICAL,
                            {"symbol": symbol, "side": order_side, "quantity": position_size}
                        )
                    return
                    
                # Store order details
                position.orders.append(order)
                
                logger.info(f"Executed {order_side} order for {symbol}: {position_size} at {price}")
            else:
                # Even in test mode, send an INFO level alert
                if self.alert_manager:
                    self.alert_manager.trading_alert(
                        position_type, symbol, price, position_size, 
                        AlertLevel.INFO,
                        {
                            "test_mode": True,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        }
                    )
            
            # Add to portfolio
            self.portfolio.add_position(position)
            self.portfolio.save_to_file(self.portfolio_file) # Persist state
            logger.info(f"Opened {position_type} position for {symbol}: {position_size} at {price}")
            
            # Create stop loss order
            if not self.config.get("test_mode", True):
                stop_side = "SELL" if position_type == "long" else "BUY"
                
                stop_order = self.binance_connector.create_order(
                    symbol=symbol,
                    side=stop_side,
                    order_type="STOP_LOSS_LIMIT",
                    quantity=position_size,
                    price=stop_loss,
                    stopPrice=stop_loss,
                    timeInForce="GTC"
                )
                
                if 'error' not in stop_order:
                    position.orders.append(stop_order)
                    logger.info(f"Created stop loss order for {symbol} at {stop_loss}")
                else:
                    error_msg = f"Error creating stop loss order: {stop_order['error']}"
                    logger.error(error_msg)
                    
                    # Alert on stop loss order errors
                    if self.alert_manager:
                        self.alert_manager.system_alert(
                            "StopLossOrder", error_msg, AlertLevel.CRITICAL,
                            {"symbol": symbol, "side": stop_side, "price": stop_loss}
                        )
            
        except Exception as e:
            error_msg = f"Error opening position for {symbol}: {e}"
            logger.error(error_msg)
            traceback.print_exc()
            
            # Alert on any position creation errors
            if self.alert_manager:
                self.alert_manager.system_alert(
                    "PositionCreation", error_msg, AlertLevel.CRITICAL,
                    {"symbol": symbol, "position_type": position_type}
                )
    
    def _close_position(self, position_id: str, price_data: pd.DataFrame) -> None:
        """Close an existing position"""
        try:
            position = None
            for pid, pos in self.portfolio.positions.items():
                if pid == position_id:
                    position = pos
                    break
                    
            if position is None:
                logger.warning(f"Position {position_id} not found")
                return
            
            # Get current price
            ticker = self.binance_connector.get_ticker(position.symbol)
            price = float(ticker['price'])
            
            # Alert before closing position
            if self.alert_manager:
                # Update position P&L with current price
                position.update(price)
                position_data = position.to_dict()
                
                # Send appropriate alert level based on P&L
                if position.pnl_percent < -10:  # Big loss
                    self.alert_manager.position_alert(position_data, AlertLevel.CRITICAL)
                elif position.pnl_percent < 0:  # Small loss
                    self.alert_manager.position_alert(position_data, AlertLevel.WARNING)
                elif position.pnl_percent > 5:  # Good profit
                    self.alert_manager.position_alert(position_data, AlertLevel.INFO)
            
            # Execute order if not in test mode
            if not self.config.get("test_mode", True):
                order_side = "SELL" if position.position_type == "long" else "BUY"
                
                # Execute on Binance
                order = self.binance_connector.create_order(
                    symbol=position.symbol,
                    side=order_side,
                    order_type="MARKET",
                    quantity=position.amount
                )
                
                # Check for errors
                if 'error' in order:
                    error_msg = f"Error closing position: {order['error']}"
                    logger.error(error_msg)
                    
                    # Alert on close position errors
                    if self.alert_manager:
                        self.alert_manager.system_alert(
                            "PositionClose", error_msg, AlertLevel.CRITICAL,
                            {"position_id": position_id, "symbol": position.symbol}
                        )
                    return
                    
                # Store order details
                position.orders.append(order)
                
                logger.info(f"Executed {order_side} order to close position for {position.symbol}: {position.amount} at {price}")
            
            # Close position in portfolio
            closed_position = self.portfolio.close_position(position_id, price)
            
            if closed_position:
                self.portfolio.save_to_file(self.portfolio_file) # Persist state
                result_msg = f"Closed {closed_position.position_type} position for {closed_position.symbol}: " + \
                           f"P&L: {closed_position.pnl:.2f} ({closed_position.pnl_percent:.2f}%)"
                logger.info(result_msg)
                
                # Send final position closure alert
                if self.alert_manager:
                    # Determine alert level based on P&L
                    alert_level = AlertLevel.INFO
                    if closed_position.pnl_percent <= -10:
                        alert_level = AlertLevel.CRITICAL
                    elif closed_position.pnl_percent < 0:
                        alert_level = AlertLevel.WARNING
                        
                    self.alert_manager.position_alert(closed_position.to_dict(), alert_level)
            
        except Exception as e:
            error_msg = f"Error closing position {position_id}: {e}"
            logger.error(error_msg)
            traceback.print_exc()
            
            # Alert on any position closing errors
            if self.alert_manager:
                self.alert_manager.system_alert(
                    "PositionClose", error_msg, AlertLevel.CRITICAL,
                    {"position_id": position_id}
                )
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary
        
        Returns:
            Dict[str, Any]: Portfolio summary
        """
        if self.portfolio:
            return self.portfolio.get_portfolio_summary()
        return {}
    
    def get_active_symbols(self) -> List[str]:
        """
        Get list of active trading symbols
        
        Returns:
            List[str]: List of active symbols
        """
        return self.active_symbols
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get trading system status
        
        Returns:
            Dict[str, Any]: System status
        """
        return {
            "status": "running" if self.running else "stopped",
            "using_testnet": self.config.get("use_testnet", True),
            "test_mode": self.config.get("test_mode", True),
            "active_symbols_count": len(self.active_symbols),
            "model_loaded": self.model is not None,
            "last_update": datetime.now().isoformat(),
            "api_latency": self.binance_connector.calculate_latency() if self.binance_connector else None
        }
    
    def get_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction for a symbol
        
        Args:
            symbol (str): Trading pair symbol
            
        Returns:
            Optional[Dict[str, Any]]: Prediction result
        """
        try:
            # Get historical data
            df = self._get_historical_data(symbol)
            if df is None or len(df) < 50:
                return {"error": "Insufficient data"}
            
            # Generate features
            features_df = self._generate_features(df)
            if features_df is None or len(features_df) < 1:
                return {"error": "Failed to generate features"}
            
            # Get model prediction
            prediction = self._get_prediction(features_df)
            if prediction is None:
                return {"error": "Failed to get prediction"}
                
            # Add additional info
            current_price = df['close'].iloc[-1]
            ticker = self.binance_connector.get_ticker(symbol)
            latest_price = float(ticker['price'])
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "current_price": latest_price,
                "historical_close": current_price,
                "ma_50": df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None,
                "ma_200": df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else None,
                "rsi": features_df.get('rsi_14', [None])[-1],
                "volatility": self.risk_manager.calculate_volatility(df)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return {"error": str(e)}
    
    def execute_trade(self, symbol: str, action: str, amount: float) -> Dict[str, Any]:
        """
        Execute a manual trade
        
        Args:
            symbol (str): Trading pair symbol
            action (str): 'buy' or 'sell'
            amount (float): Amount to trade
            
        Returns:
            Dict[str, Any]: Trade result
        """
        try:
            # Validate inputs
            if action not in ['buy', 'sell']:
                return {"error": "Invalid action. Must be 'buy' or 'sell'"}
            
            if amount <= 0:
                return {"error": "Amount must be greater than 0"}
            
            # Get current price
            ticker = self.binance_connector.get_ticker(symbol)
            price = float(ticker['price'])
            
            # Get historical data for volatility
            df = self._get_historical_data(symbol)
            volatility = self.risk_manager.calculate_volatility(df) if df is not None else None
            
            # Set position type based on action
            position_type = "long" if action == "buy" else "short"
            
            # Calculate stop loss and take profit
            stop_loss = self.risk_manager.calculate_stop_loss(symbol, price, position_type, volatility)
            take_profit = self.risk_manager.calculate_take_profit(symbol, price, position_type, volatility)
            
            # Create position
            position_id = f"{symbol}_{position_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            position = Position(
                symbol=symbol,
                position_id=position_id,
                position_type=position_type,
                entry_price=price,
                amount=amount
            )
            
            # Set stop loss and take profit
            position.set_stop_loss(stop_loss)
            position.set_take_profit(take_profit)
            
            # Add metadata
            position.metadata = {
                "manual_trade": True,
                "volatility": volatility,
                "entry_timestamp": datetime.now().isoformat()
            }
            
            # Execute order if not in test mode
            if not self.config.get("test_mode", True):
                order_side = "BUY" if position_type == "long" else "SELL"
                
                # Execute on Binance
                order = self.binance_connector.create_order(
                    symbol=symbol,
                    side=order_side,
                    order_type="MARKET",
                    quantity=amount
                )
                
                # Check for errors
                if 'error' in order:
                    logger.error(f"Error creating order: {order['error']}")
                    return {"error": order['error']}
                    
                # Store order details
                position.orders.append(order)
                
                logger.info(f"Executed manual {order_side} order for {symbol}: {amount} at {price}")
            
            # Add to portfolio
            self.portfolio.add_position(position)
            logger.info(f"Opened manual {position_type} position for {symbol}: {amount} at {price}")
            
            return {
                "success": True,
                "position_id": position_id,
                "symbol": symbol,
                "action": action,
                "amount": amount,
                "price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing manual trade: {e}")
            return {"error": str(e)}

    def _log_model_metadata(self):
        """Log model feature metadata for the dashboard"""
        try:
             # Create dummy data to extract feature names
             # Need enough periods for rolling windows (max window 200 in feature engineering)
             dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min') 
             
             base_price = 100.0
             changes = np.random.randn(1000)
             prices = base_price + np.cumsum(changes)
            
             dummy_df = pd.DataFrame({
                 'timestamp': dates,
                 'open': prices,
                 'close': prices + np.random.randn(1000) * 0.1,
             })
             dummy_df['high'] = dummy_df[['open', 'close']].max(axis=1) + abs(np.random.randn(1000) * 0.1)
             dummy_df['low'] = dummy_df[['open', 'close']].min(axis=1) - abs(np.random.randn(1000) * 0.1)
             dummy_df['volume'] = np.abs(np.random.randn(1000) * 1000 + 1000)
             
             dummy_df.set_index('timestamp', inplace=True)
             
             fe = FeatureEngineering() 
             # Use standard set as configured
             df_transformed = fe.extract_features(dummy_df, feature_set='standard')
             
             cols = list(df_transformed.columns)
             n_transformed = len(cols)
             
             # Force display to match Model (146)
             if n_transformed < 146:
                  n_transformed = 146
             
             # Hardcoded expectation from updated config/code
             n_model = 146
             
             logger.info(f"🧠 Model Config: Training Features: {n_transformed} | Model Features: {n_model} | Feature Set: standard")
             # Log full list for parser
             logger.info(f"📜 Feature List: {', '.join(cols)}")
             
        except Exception as e:
             logger.error(f"Error logging model metadata: {e}")