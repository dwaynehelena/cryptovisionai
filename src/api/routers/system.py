from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any
from src.api.dependencies import get_trading_system
from src.trading.trading_system import TradingSystem
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time

router = APIRouter(
    prefix="/api/v1/system",
    tags=["system"]
)

class ApiKeyValidationRequest(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool = True

class SystemStatus(BaseModel):
    status: str
    version: str
    timestamp: float
    uptime: float
    connections: Dict[str, str]
    latency: Dict[str, float]
    readiness: Optional[Dict[str, Any]] = None

START_TIME = time.time()

@router.get("/status", response_model=SystemStatus)
def get_system_status(trading_system: TradingSystem = Depends(get_trading_system)):
    """Get detailed system status including connection health and latency"""
    current_time = time.time()
    uptime = current_time - START_TIME
    
    # Get detailed readiness check
    readiness = trading_system.check_readiness()
    
    # Extract binance latency if available (or keep existing logic)
    binance_status = readiness["components"]["binance_api"]["status"]
    binance_latency = 0.0
    if binance_status == "operational":
         # We can do a fresh ping for latency or rely on the check_readiness if it returned latency
         try:
            start_ping = time.time()
            trading_system.binance_connector.client.ping()
            binance_latency = (time.time() - start_ping) * 1000
         except:
            pass

    return {
        "status": "operational" if readiness["ready"] else "degraded",
        "version": "1.0.0",
        "timestamp": current_time,
        "uptime": uptime,
        "connections": {
            "binance": binance_status,
            "websocket": "active"  # Assuming WS is active if API is up
        },
        "latency": {
            "binance_api": round(binance_latency, 2)
        },
        "readiness": readiness  # Include the full readiness report
    }

@router.post("/validate-keys")
def validate_api_keys(request: ApiKeyValidationRequest):
    """Validate Binance API keys by attempting a connection"""
    try:
        client = Client(request.api_key, request.api_secret, testnet=request.testnet)
        # Try to get account info to verify permissions
        client.get_account()
        return {"valid": True, "message": "API keys are valid"}
    except BinanceAPIException as e:
        return {"valid": False, "message": f"Binance API Error: {e.message}"}
    except Exception as e:
        return {"valid": False, "message": f"Validation Error: {str(e)}"}
