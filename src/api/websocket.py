from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Store connections by symbol
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        async with self.lock:
            if symbol not in self.active_connections:
                self.active_connections[symbol] = set()
            self.active_connections[symbol].add(websocket)
        logger.info(f"Client connected to {symbol}. Total: {len(self.active_connections[symbol])}")
    
    async def disconnect(self, websocket: WebSocket, symbol: str):
        """Remove a WebSocket connection"""
        async with self.lock:
            if symbol in self.active_connections:
                self.active_connections[symbol].discard(websocket)
                if len(self.active_connections[symbol]) == 0:
                    del self.active_connections[symbol]
        logger.info(f"Client disconnected from {symbol}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict, symbol: str):
        """Broadcast a message to all clients subscribed to a symbol"""
        if symbol not in self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections[symbol]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        async with self.lock:
            for connection in disconnected:
                self.active_connections[symbol].discard(connection)

# Global connection manager instance
manager = ConnectionManager()

async def market_data_stream(websocket: WebSocket, symbol: str, binance_connector):
    """Stream market data to a WebSocket client"""
    await manager.connect(websocket, symbol)
    
    try:
        while True:
            # Fetch latest price data
            ticker = binance_connector.get_ticker(symbol)
            
            if 'error' not in ticker:
                message = {
                    "type": "price_update",
                    "symbol": symbol,
                    "price": ticker.get('price'),
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(message, websocket)
            
            # Wait before next update
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket, symbol)
    except Exception as e:
        logger.error(f"Error in market data stream: {e}")
        await manager.disconnect(websocket, symbol)
