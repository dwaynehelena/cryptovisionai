from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import market, portfolio, orders, risk, system, analysis
from src.api.config import settings
from src.api.middleware import catch_exceptions_middleware, log_requests_middleware, setup_exception_handlers
from src.api.middleware.cache import CacheMiddleware
from src.api.websocket import market_data_stream, manager
from src.api.dependencies import set_trading_system
from src.trading.trading_system import TradingSystem
import logging
import traceback
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global trading system instance
trading_system_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    global trading_system_instance
    logger.info("Starting up CryptoVisionAI API...")
    
    # Initialize trading system
    trading_config = {
        "use_testnet": settings.BINANCE_USE_TESTNET,
        "initial_capital": settings.TRADING_INITIAL_CAPITAL,
        "risk_management": {
            "max_position_size": settings.TRADING_MAX_POSITION_SIZE,
            "max_open_positions": settings.TRADING_MAX_OPEN_POSITIONS,
            "stop_loss_percent": settings.TRADING_STOP_LOSS_PERCENT,
            "take_profit_percent": settings.TRADING_TAKE_PROFIT_PERCENT,
        },
        "model_path": settings.MODEL_PATH,
        "api_key": settings.BINANCE_API_KEY,
        "api_secret": settings.BINANCE_API_SECRET,
    }
    
    try:
        logger.info(f"Initializing TradingSystem with config: testnet={trading_config['use_testnet']}, model={trading_config['model_path']}")
        trading_system_instance = TradingSystem(trading_config)
        set_trading_system(trading_system_instance)
        logger.info("Trading system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trading system: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CryptoVisionAI API...")
    if trading_system_instance:
        trading_system_instance.stop()

# Create FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        lifespan=lifespan
    )

    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(CacheMiddleware, ttl=60)

    # Add custom middleware
    app.middleware("http")(log_requests_middleware)
    app.middleware("http")(catch_exceptions_middleware)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Include routers
    app.include_router(market.router)
    app.include_router(portfolio.router)
    app.include_router(orders.router)
    app.include_router(risk.router)
    app.include_router(system.router)
    app.include_router(analysis.router)
    
    logger.info("FastAPI application started successfully")
    
    return app

# Create FastAPI app
app = create_app()

@app.get("/")
def root():
    return {
        "message": "Welcome to CryptoVisionAI API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": settings.VERSION,
        "trading_system": "initialized" if trading_system_instance else "not initialized"
    }

@app.websocket("/ws/market/{symbol}")
async def websocket_market_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data"""
    if not trading_system_instance:
        await websocket.close(code=1000, reason="Trading system not initialized")
        return
    
    await market_data_stream(
        websocket,
        symbol,
        trading_system_instance.binance_connector
    )
