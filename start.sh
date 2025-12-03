#!/bin/bash

# CryptoVisionAI Startup Script

echo "🚀 Starting CryptoVisionAI Platform..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your Binance API credentials"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Start backend API
echo "🐍 Starting FastAPI backend..."
cd /Users/helenadw/cryptovisonai
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "⚛️  Starting React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ CryptoVisionAI is running!"
echo "📊 Frontend: http://localhost:5173"
echo "🔌 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C to kill both processes
trap "echo '🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Wait for both processes
wait
