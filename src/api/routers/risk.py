from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.trading.trading_system import TradingSystem
from src.api.dependencies import get_trading_system
from typing import List, Dict, Optional

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


class RiskLimitsUpdate(BaseModel):
    max_position_size_pct: Optional[float] = None
    max_risk_per_trade_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    daily_loss_limit_pct: Optional[float] = None
    auto_stop_loss_enabled: Optional[bool] = None
    atr_multiplier: Optional[float] = None
    max_correlation: Optional[float] = None


class PositionSizeRequest(BaseModel):
    entry_price: float
    stop_price: float
    volatility: Optional[float] = None


class EmergencyCloseRequest(BaseModel):
    confirmation: str  # Must be "CONFIRM_CLOSE_ALL"


@router.get("/limits")
def get_risk_limits(trading_system: TradingSystem = Depends(get_trading_system)):
    """Get current risk limits"""
    try:
        risk_mgr = trading_system.risk_manager
        limits = risk_mgr.limits
        
        return {
            "max_position_size_pct": limits.max_position_size_pct,
            "max_risk_per_trade_pct": limits.max_risk_per_trade_pct,
            "max_drawdown_pct": limits.max_drawdown_pct,
            "daily_loss_limit_pct": limits.daily_loss_limit_pct,
            "auto_stop_loss_enabled": limits.auto_stop_loss_enabled,
            "atr_multiplier": limits.atr_multiplier,
            "max_correlation": limits.max_correlation,
            "trading_enabled": limits.trading_enabled
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/limits")
def update_risk_limits(
    limits: RiskLimitsUpdate,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Update risk limits"""
    try:
        risk_mgr = trading_system.risk_manager
        
        # Update only provided fields
        update_dict = {k: v for k, v in limits.dict().items() if v is not None}
        risk_mgr.update_limits(**update_dict)
        
        return {
            "success": True,
            "message": "Risk limits updated successfully",
            "limits": {
                "max_position_size_pct": risk_mgr.limits.max_position_size_pct,
                "max_risk_per_trade_pct": risk_mgr.limits.max_risk_per_trade_pct,
                "max_drawdown_pct": risk_mgr.limits.max_drawdown_pct,
                "daily_loss_limit_pct": risk_mgr.limits.daily_loss_limit_pct,
                "auto_stop_loss_enabled": risk_mgr.limits.auto_stop_loss_enabled,
                "atr_multiplier": risk_mgr.limits.atr_multiplier,
                "max_correlation": risk_mgr.limits.max_correlation
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def get_risk_status(trading_system: TradingSystem = Depends(get_trading_system)):
    """Get current risk status"""
    try:
        portfolio = trading_system.portfolio
        risk_mgr = trading_system.risk_manager
        
        # Get current portfolio value
        current_value = portfolio.total_value
        
        # Get position values
        position_values = {}
        for symbol, position in portfolio.positions.items():
            position_values[symbol] = position.quantity * position.current_price
        
        # Get risk status
        status = risk_mgr.get_status(current_value, position_values)
        
        return {
            "current_drawdown": round(status.current_drawdown, 2),
            "peak_value": round(status.peak_value, 2),
            "daily_pnl": round(status.daily_pnl, 2),
            "capital_used": round(status.capital_used, 2),
            "capital_available": round(status.capital_available, 2),
            "largest_position_pct": round(status.largest_position_pct, 2),
            "risk_score": status.risk_score,
            "trading_enabled": status.trading_enabled,
            "violations": status.violations,
            "last_updated": status.last_updated.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-position-size")
def calculate_position_size(
    request: PositionSizeRequest,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Calculate recommended position size based on risk parameters"""
    try:
        portfolio = trading_system.portfolio
        risk_mgr = trading_system.risk_manager
        
        position_size = risk_mgr.calculate_position_size(
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            portfolio_value=portfolio.total_value,
            volatility=request.volatility
        )
        
        position_value = position_size * request.entry_price
        risk_amount = position_size * abs(request.entry_price - request.stop_price)
        risk_pct = (risk_amount / portfolio.total_value) * 100
        
        return {
            "recommended_size": round(position_size, 4),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round(risk_pct, 2),
            "entry_price": request.entry_price,
            "stop_price": request.stop_price
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/violations")
def get_risk_violations(
    limit: int = 20,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Get recent risk violations"""
    try:
        risk_mgr = trading_system.risk_manager
        
        violations = [
            {
                "timestamp": v[0].isoformat(),
                "type": v[1].value,
                "message": v[2]
            }
            for v in risk_mgr.violations[-limit:]
        ]
        
        return {
            "violations": violations,
            "total_count": len(risk_mgr.violations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-close-all")
def emergency_close_all(
    request: EmergencyCloseRequest,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Emergency close all positions"""
    try:
        # Verify confirmation
        if request.confirmation != "CONFIRM_CLOSE_ALL":
            raise HTTPException(
                status_code=400,
                detail="Invalid confirmation. Must be 'CONFIRM_CLOSE_ALL'"
            )
        
        portfolio = trading_system.portfolio
        closed_positions = []
        errors = []
        
        # Close all positions
        for symbol, position in list(portfolio.positions.items()):
            try:
                # Place market sell order
                result = trading_system.binance_connector.client.create_order(
                    symbol=symbol,
                    side='SELL' if position.side == 'LONG' else 'BUY',
                    type='MARKET',
                    quantity=position.quantity
                )
                
                closed_positions.append({
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "order_id": result['orderId']
                })
                
                # Remove from portfolio
                del portfolio.positions[symbol]
                
            except Exception as e:
                errors.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        # Disable trading
        trading_system.risk_manager.disable_trading()
        
        return {
            "success": True,
            "message": f"Emergency close executed. Closed {len(closed_positions)} positions.",
            "closed_positions": closed_positions,
            "errors": errors,
            "trading_disabled": True
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable-trading")
def enable_trading(trading_system: TradingSystem = Depends(get_trading_system)):
    """Re-enable trading after manual review"""
    try:
        trading_system.risk_manager.enable_trading()
        
        return {
            "success": True,
            "message": "Trading enabled",
            "trading_enabled": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable-trading")
def disable_trading(trading_system: TradingSystem = Depends(get_trading_system)):
    """Disable trading"""
    try:
        trading_system.risk_manager.disable_trading()
        
        return {
            "success": True,
            "message": "Trading disabled",
            "trading_enabled": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation-matrix")
def get_correlation_matrix(trading_system: TradingSystem = Depends(get_trading_system)):
    """Get correlation matrix between open positions"""
    try:
        # Get historical data for each position
        positions_data = {}
        
        for symbol in trading_system.portfolio.positions.keys():
            try:
                df = trading_system.binance_connector.get_historical_klines(symbol, '1h', 100)
                if df is not None and not df.empty:
                    positions_data[symbol] = df
            except:
                continue
        
        if len(positions_data) < 2:
            return {
                "correlation_matrix": {},
                "high_correlations": [],
                "message": "Need at least 2 positions to calculate correlation"
            }
        
        # Calculate correlation
        corr_matrix = trading_system.risk_manager.calculate_correlation(positions_data)
        
        # Find high correlations
        high_corr = trading_system.risk_manager.check_correlation_risk(corr_matrix)
        
        # Convert to dict for JSON
        corr_dict = corr_matrix.to_dict() if not corr_matrix.empty else {}
        
        return {
            "correlation_matrix": corr_dict,
            "high_correlations": [
                {
                    "symbol1": pair[0],
                    "symbol2": pair[1],
                    "correlation": round(pair[2], 3)
                }
                for pair in high_corr
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
