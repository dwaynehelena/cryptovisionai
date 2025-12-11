from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.trading.trading_system import TradingSystem
from src.api.dependencies import get_trading_system
from typing import Optional, List
from enum import Enum

router = APIRouter(prefix="/api/v1/orders", tags=["orders"])

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderRequest(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None

@router.post("/market")
def place_market_order(
    order: OrderRequest,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Place a market order"""
    try:
        # Validate order
        if order.order_type != OrderType.MARKET:
            raise HTTPException(status_code=400, detail="Invalid order type for market order")
        
        # Place order through Binance
        result = trading_system.binance_connector.client.create_order(
            symbol=order.symbol,
            side=order.side.value,
            type="MARKET",
            quantity=order.quantity
        )
        
        return {
            "success": True,
            "order_id": result['orderId'],
            "symbol": result['symbol'],
            "side": result['side'],
            "quantity": result['executedQty'],
            "price": result.get('price', 'MARKET'),
            "status": result['status']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limit")
def place_limit_order(
    order: OrderRequest,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Place a limit order"""
    try:
        if order.order_type != OrderType.LIMIT:
            raise HTTPException(status_code=400, detail="Invalid order type for limit order")
        
        if not order.price:
            raise HTTPException(status_code=400, detail="Price required for limit order")
        
        result = trading_system.binance_connector.client.create_order(
            symbol=order.symbol,
            side=order.side.value,
            type="LIMIT",
            timeInForce="GTC",
            quantity=order.quantity,
            price=str(order.price)
        )
        
        return {
            "success": True,
            "order_id": result['orderId'],
            "symbol": result['symbol'],
            "side": result['side'],
            "quantity": result['origQty'],
            "price": result['price'],
            "status": result['status']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
def place_stop_order(
    order: OrderRequest,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Place a stop-loss order"""
    try:
        if not order.stop_price:
            raise HTTPException(status_code=400, detail="Stop price required")
        
        result = trading_system.binance_connector.client.create_order(
            symbol=order.symbol,
            side=order.side.value,
            type="STOP_LOSS_LIMIT",
            timeInForce="GTC",
            quantity=order.quantity,
            price=str(order.price or order.stop_price),
            stopPrice=str(order.stop_price)
        )
        
        return {
            "success": True,
            "order_id": result['orderId'],
            "symbol": result['symbol'],
            "side": result['side'],
            "quantity": result['origQty'],
            "stop_price": result['stopPrice'],
            "status": result['status']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active")
def get_active_orders(
    symbol: Optional[str] = None,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Get all active (open) orders"""
    try:
        if symbol:
            orders = trading_system.binance_connector.get_open_orders(symbol=symbol)
        else:
            orders = trading_system.binance_connector.get_open_orders()
        
        # Check if error returned (legacy behavior handling, though updated connector returns [])
        if orders and isinstance(orders[0], dict) and 'error' in orders[0]:
            return []

        return [{
            "order_id": order['orderId'],
            "symbol": order['symbol'],
            "side": order['side'],
            "type": order['type'],
            "quantity": order['origQty'],
            "price": order.get('price'),
            "stop_price": order.get('stopPrice'),
            "status": order['status'],
            "time": order['time']
        } for order in orders]
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
def get_order_history(
    symbol: str,
    limit: int = 50,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Get order history"""
    try:
        orders = trading_system.binance_connector.client.get_all_orders(
            symbol=symbol,
            limit=limit
        )
        
        return [{
            "order_id": order['orderId'],
            "symbol": order['symbol'],
            "side": order['side'],
            "type": order['type'],
            "quantity": order['origQty'],
            "executed_qty": order['executedQty'],
            "price": order.get('price'),
            "status": order['status'],
            "time": order['time']
        } for order in orders]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{order_id}")
def cancel_order(
    order_id: int,
    symbol: str,
    trading_system: TradingSystem = Depends(get_trading_system)
):
    """Cancel an active order"""
    try:
        result = trading_system.binance_connector.client.cancel_order(
            symbol=symbol,
            orderId=order_id
        )
        
        return {
            "success": True,
            "order_id": result['orderId'],
            "symbol": result['symbol'],
            "status": result['status']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/calculate-risk")
def calculate_risk(
    entry_price: float,
    stop_loss: float,
    account_balance: float,
    risk_percent: float = 2.0
):
    """Calculate position size based on risk parameters"""
    try:
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate maximum risk amount
        max_risk_amount = account_balance * (risk_percent / 100)
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_share
        
        # Calculate position value
        position_value = position_size * entry_price
        
        # Calculate risk/reward for various targets
        risk_amount = position_size * risk_per_share
        
        return {
            "position_size": round(position_size, 4),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": risk_percent,
            "risk_per_share": round(risk_per_share, 2),
            "reward_ratios": {
                "1:1": round(entry_price + risk_per_share, 2),
                "1:2": round(entry_price + (risk_per_share * 2), 2),
                "1:3": round(entry_price + (risk_per_share * 3), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
