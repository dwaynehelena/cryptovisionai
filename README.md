# CryptoVisionAI 🚀

**CryptoVisionAI** is a production-ready, full-stack cryptocurrency trading and analysis platform. It combines real-time market data, advanced technical analysis, AI-driven insights, and professional trading tools into a single, customizable dashboard.

![Dashboard Preview](https://via.placeholder.com/800x400?text=CryptoVisionAI+Dashboard)

## ✨ Features

- **Real-Time Data**: Live WebSocket feeds for prices, order books, and trades.
- **Advanced Charts**: Interactive charts with RSI, MACD, Bollinger Bands, and more.
- **AI & Sentiment**: Machine learning price predictions and social sentiment analysis.
- **Trading Suite**: Limit, Market, Stop-Loss, and Take-Profit orders with risk management.
- **Portfolio Sync**: Real-time tracking of your Binance portfolio performance.
- **Customizable UI**: Drag-and-drop dashboard, dark/light themes, and keyboard shortcuts.
- **Strategy Lab**: Backtest trading strategies against historical data.

## 🛠️ Tech Stack

- **Frontend**: React, TypeScript, Material-UI, Recharts, React Grid Layout.
- **Backend**: Python, FastAPI, Uvicorn, Pandas, TA-Lib.
- **Data**: Binance API (Spot & Futures).

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- Binance Account (Testnet recommended for development)

### Quick Start (Docker)

The easiest way to run the application is using Docker Compose.

```bash
# 1. Clone the repository
git clone https://github.com/dwaynehelena/cryptovisionai.git
cd cryptovisionai

# 2. Configure Environment
cp .env.example .env
# Edit .env with your Binance API credentials

# 3. Run with Docker Compose
docker-compose up --build
```

Access the application at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000/docs

### Manual Installation

#### Backend

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
# Note: TA-Lib must be installed on your system first (brew install ta-lib on Mac)
pip install -r requirements.txt

# 3. Run the server
uvicorn src.api.main:app --reload
```

#### Frontend

```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Run development server
npm run dev
```

## 📖 Documentation

- **[Release Notes](RELEASE_NOTES.md)**: See what's new in v1.0.0.
- **[Task List](task.md)**: View the development roadmap and completed features.

## ⌨️ Keyboard Shortcuts

- `Shift + D`: Toggle Dark/Light Theme
- `Shift + ?`: Show Help
- `Shift + B`: Focus Buy Order
- `Shift + S`: Focus Sell Order

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not risk money you cannot afford to lose. The authors are not responsible for any financial losses incurred while using this software.

## 📄 License

MIT License
