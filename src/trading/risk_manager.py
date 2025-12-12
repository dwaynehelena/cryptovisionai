"""
Risk Manager for CryptoVisionAI
Handles position sizing, risk limits, drawdown protection, and automated risk controls.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum


class RiskViolationType(Enum):
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    DRAWDOWN_LIMIT_REACHED = "drawdown_limit_reached"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    HIGH_CORRELATION = "high_correlation"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class RiskLimits:
    """Risk management parameters"""
    max_position_size_pct: float = 10.0  # % of portfolio per position
    max_risk_per_trade_pct: float = 2.0  # % of portfolio at risk per trade
    max_drawdown_pct: float = 20.0  # Maximum drawdown from peak
    daily_loss_limit_pct: float = 5.0  # Maximum daily loss
    auto_stop_loss_enabled: bool = False
    atr_multiplier: float = 2.0  # ATR multiplier for stop-loss
    max_correlation: float = 0.8  # Maximum correlation between positions
    trading_enabled: bool = True


@dataclass
class RiskStatus:
    """Current risk status"""
    current_drawdown: float
    peak_value: float
    daily_pnl: float
    capital_used: float
    capital_available: float
    largest_position_pct: float
    risk_score: int  # 0-100
    trading_enabled: bool
    violations: List[str]
    last_updated: datetime


class RiskManager:
    """Manages risk controls and validates trading decisions"""
    
    def __init__(self, initial_capital: float):
        self.limits = RiskLimits()
        self.initial_capital = initial_capital
        self.peak_value = initial_capital
        self.daily_pnl_start = datetime.utcnow().date()
        self.daily_pnl = 0.0
        self.violations: List[Tuple[datetime, RiskViolationType, str]] = []
        
    def update_limits(self, **kwargs):
        """Update risk limits"""
        for key, value in kwargs.items():
            if hasattr(self.limits, key):
                setattr(self.limits, key, value)
    
    def validate_position_size(
        self,
        symbol: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        existing_position_value: float = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if position size is within limits
        
        Returns:
            (is_valid, error_message)
        """
        position_value = quantity * price
        total_position_value = existing_position_value + position_value
        position_pct = (total_position_value / portfolio_value) * 100
        
        max_allowed_pct = self.limits.max_position_size_pct
        
        if position_pct > max_allowed_pct:
            error = f"Position size {position_pct:.1f}% exceeds limit of {max_allowed_pct:.1f}%"
            self._log_violation(RiskViolationType.POSITION_SIZE_EXCEEDED, error)
            return False, error
        
        return True, None
    
    def _calculate_position_size_core(
        self,
        entry_price: float,
        stop_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            entry_price: Entry price for the position
            stop_price: Stop-loss price
            portfolio_value: Current portfolio value
            volatility: Optional volatility adjustment (0-1)
        
        Returns:
            Recommended position size
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        # Calculate maximum risk amount
        max_risk_amount = portfolio_value * (self.limits.max_risk_per_trade_pct / 100)
        
        # Base position size
        position_size = max_risk_amount / risk_per_share if risk_per_share > 0 else 0
        
        # Adjust for volatility if provided
        if volatility is not None and volatility > 0:
            volatility_adjustment = 1 / (1 + volatility)
            position_size *= volatility_adjustment
        
        # Ensure position doesn't exceed max position size
        max_position_value = portfolio_value * (self.limits.max_position_size_pct / 100)
        max_size_by_limit = max_position_value / entry_price
        
        return min(position_size, max_size_by_limit)

    def calculate_position_size(self, symbol: str, price: float, signal_strength: float, 
                               volatility: Optional[float] = None) -> float:
        """
        Wrapper to match TradingSystem signature
        """
        # We need a stop price to calculate risk-based sizing
        # Assume LONG for strictly sizing purposes (distance is absolute anyway)
        stop_price = self.calculate_stop_loss(symbol, price, "LONG", volatility)
        
        # Logic to upscale size based on signal_strength (optional, referencing original internal RiskManager)
        # Original: position_value *= (0.5 + signal_strength)
        
        base_size = self._calculate_position_size_core(price, stop_price, self.peak_value, volatility)
        
        # Apply signal strength multiplier (0.5 to 1.5)
        # Default signal_strength is often around 0.5-0.9?
        # If signal_strength is 0-1, we can map it.
        if signal_strength > 0:
             base_size *= (0.5 + signal_strength)
             
        return base_size
    
    def calculate_atr_stop(
        self,
        symbol_data: pd.DataFrame,
        entry_price: float,
        side: str = "LONG",
        period: int = 14
    ) -> float:
        """
        Calculate ATR-based stop-loss price
        
        Args:
            symbol_data: DataFrame with 'high', 'low', 'close' columns
            entry_price: Entry price
            side: "LONG" or "SHORT"
            period: ATR period
        
        Returns:
            Stop-loss price
        """
        # Calculate True Range
        high_low = symbol_data['high'] - symbol_data['low']
        high_close = abs(symbol_data['high'] - symbol_data['close'].shift())
        low_close = abs(symbol_data['low'] - symbol_data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        stop_distance = atr * self.limits.atr_multiplier
        
        if side == "LONG":
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance
    
    def check_drawdown(self, current_value: float) -> Tuple[bool, float, bool]:
        """
        Check if drawdown limits are breached
        
        Returns:
            (limit_breached, current_drawdown_pct, should_halt)
        """
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate drawdown
        drawdown = ((current_value - self.peak_value) / self.peak_value) * 100
        
        # Check if limit is breached
        limit_breached = abs(drawdown) >= self.limits.max_drawdown_pct
        
        if limit_breached:
            error = f"Drawdown {abs(drawdown):.1f}% exceeds limit of {self.limits.max_drawdown_pct}%"
            self._log_violation(RiskViolationType.DRAWDOWN_LIMIT_REACHED, error)
            self.limits.trading_enabled = False
        
        return limit_breached, drawdown, not self.limits.trading_enabled
    
    def check_daily_loss(self, current_pnl: float) -> Tuple[bool, str]:
        """
        Check if daily loss limit is exceeded
        
        Returns:
            (limit_exceeded, message)
        """
        today = datetime.utcnow().date()
        
        # Reset daily P&L if it's a new day
        if today > self.daily_pnl_start:
            self.daily_pnl = 0.0
            self.daily_pnl_start = today
        
        self.daily_pnl = current_pnl
        
        # Check limit (only for losses)
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl / self.peak_value) * 100
            
            if daily_loss_pct >= self.limits.daily_loss_limit_pct:
                error = f"Daily loss {daily_loss_pct:.1f}% exceeds limit of {self.limits.daily_loss_limit_pct}%"
                self._log_violation(RiskViolationType.DAILY_LOSS_LIMIT, error)
                return True, error
        
        return False, ""
    
    def calculate_correlation(
        self,
        positions: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between positions
        
        Args:
            positions: Dict of symbol -> price DataFrame
        
        Returns:
            Correlation matrix
        """
        if len(positions) < 2:
            return pd.DataFrame()
        
        # Combine all price series
        price_data = {}
        for symbol, df in positions.items():
            if 'close' in df.columns:
                price_data[symbol] = df['close']
        
        if not price_data:
            return pd.DataFrame()
        
        # Create DataFrame and calculate returns
        prices_df = pd.DataFrame(price_data)
        returns = prices_df.pct_change().dropna()
        
        # Calculate correlation
        return returns.corr()
    
    def check_correlation_risk(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """
        Check for high correlation between positions
        
        Returns:
            List of (symbol1, symbol2, correlation) tuples exceeding threshold
        """
        high_correlations = []
        
        if correlation_matrix.empty:
            return high_correlations
        
        # Find pairs with high correlation
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= self.limits.max_correlation:
                    symbol1 = correlation_matrix.index[i]
                    symbol2 = correlation_matrix.index[j]
                    high_correlations.append((symbol1, symbol2, corr))
                    
                    error = f"High correlation {corr:.2f} between {symbol1} and {symbol2}"
                    self._log_violation(RiskViolationType.HIGH_CORRELATION, error)
        
        return high_correlations
    
    def calculate_risk_score(
        self,
        current_value: float,
        positions: Dict[str, float],  # symbol -> value
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Calculate overall risk score (0-100)
        
        Higher score = Higher risk
        """
        score = 0
        
        # Factor 1: Drawdown (30 points max)
        drawdown_pct = abs(((current_value - self.peak_value) / self.peak_value) * 100)
        score += min((drawdown_pct / self.limits.max_drawdown_pct) * 30, 30)
        
        # Factor 2: Position concentration (25 points max)
        if positions and current_value > 0:
            largest_position = max(positions.values()) if positions else 0
            concentration_pct = (largest_position / current_value) * 100
            score += min((concentration_pct / 100) * 25, 25)
        
        # Factor 3: Capital utilization (20 points max)
        if positions and current_value > 0:
            total_position_value = sum(positions.values())
            utilization = total_position_value / current_value
            if utilization > 1:  # Leverage
                score += min((utilization - 1) * 20, 20)
        
        # Factor 4: Correlation (15 points max)
        if correlation_matrix is not None and not correlation_matrix.empty:
            # Average absolute correlation
            avg_corr = correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            score += avg_corr * 15
        
        # Factor 5: Daily loss (10 points max)
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl / self.peak_value) * 100
            score += min((daily_loss_pct / self.limits.daily_loss_limit_pct) * 10, 10)
        
        return int(min(score, 100))
    
    def get_status(
        self,
        current_value: float,
        positions: Dict[str, float]
    ) -> RiskStatus:
        """Get current risk status"""
        drawdown_pct = ((current_value - self.peak_value) / self.peak_value) * 100
        
        total_position_value = sum(positions.values()) if positions else 0
        capital_used = total_position_value
        capital_available = max(current_value - capital_used, 0)
        
        largest_position = max(positions.values()) if positions else 0
        largest_position_pct = (largest_position / current_value * 100) if current_value > 0 else 0
        
        risk_score = self.calculate_risk_score(current_value, positions)
        
        # Get recent violations
        recent_violations = [
            f"{v[1].value}: {v[2]}"
            for v in self.violations[-5:]  # Last 5 violations
        ]
        
        return RiskStatus(
            current_drawdown=drawdown_pct,
            peak_value=self.peak_value,
            daily_pnl=self.daily_pnl,
            capital_used=capital_used,
            capital_available=capital_available,
            largest_position_pct=largest_position_pct,
            risk_score=risk_score,
            trading_enabled=self.limits.trading_enabled,
            violations=recent_violations,
            last_updated=datetime.utcnow()
        )
    
    def _log_violation(self, violation_type: RiskViolationType, message: str):
        """Log a risk violation"""
        self.violations.append((datetime.utcnow(), violation_type, message))
        
        # Keep only last 100 violations
        if len(self.violations) > 100:
            self.violations = self.violations[-100:]
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.daily_pnl_start = datetime.utcnow().date()
    
    def enable_trading(self):
        """Re-enable trading after manual review"""
        self.limits.trading_enabled = True
    
    def disable_trading(self):
        """Disable trading"""
        self.limits.trading_enabled = False

    # --- Compatibility Methods for TradingSystem ---

    def calculate_volatility(self, price_data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate asset volatility (standard deviation of returns)
        """
        if len(price_data) < 2:
            return 0.0
            
        returns = price_data['close'].pct_change().dropna()
        if len(returns) > window:
            returns = returns[-window:]
            
        return float(returns.std())

    def calculate_stop_loss(self, symbol: str, price: float, position_type: str, 
                           volatility: Optional[float] = None) -> float:
        """
        Calculate stop loss price based on risk parameters and volatility
        """
        # Base stop loss on risk limit or volatility
        # Use ATR-like logic if volatility provided, else fixed %
        
        stop_percent = self.limits.max_risk_per_trade_pct # Default use trade risk % as distance logic base
        
        if volatility is not None and volatility > 0:
            # If volatility is provided, use it to scale the stop
            # Volatility is std dev of returns. 2 std devs is a good stop.
            stop_percent = volatility * 2.0 * 100
            
        # Hard cap
        stop_percent = min(stop_percent, 10.0)
        
        if position_type.lower() == "long":
            stop_price = price * (1 - stop_percent / 100)
        else: # short
            stop_price = price * (1 + stop_percent / 100)
            
        return stop_price

    def calculate_take_profit(self, symbol: str, price: float, position_type: str,
                             volatility: Optional[float] = None) -> float:
        """
        Calculate take profit price based on risk parameters
        """
        # Usually TP is 2x or 3x risk (Risk/Reward ratio)
        # We can calculate SL distance first
        stop_price = self.calculate_stop_loss(symbol, price, position_type, volatility)
        risk_distance = abs(price - stop_price)
        
        reward_distance = risk_distance * 2.0 # 1:2 Risk/Reward
        
        if position_type.lower() == "long":
            tp_price = price + reward_distance
        else: # short
            tp_price = price - reward_distance
            
        return tp_price

    def can_open_position(self, symbol: str, price_data: pd.DataFrame) -> bool:
        """
        Check if a new position can be opened based on risk limits
        """
        # Basic check: is trading enabled?
        if not self.limits.trading_enabled:
            return False
            
        # We could check validation logic here if we had portfolio state
        # For now, we assume if trading is enabled and no violations, it is OK.
        # Specific position sizing limits are checked in 'calculate_position_size' / 'validate_position_size'
        
        return True
