#!/usr/bin/env python3
"""
Dashboard Module - Provides real-time visualization of market data, predictions and trading performance
"""

import dash
from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Union
import os
import json
from flask import jsonify

# Import custom modules
from src.trading.trading_system import TradingSystem
from src.data_processing.binance_connector import BinanceConnector
from src.monitoring import dashboard_monitor

# Configure logging
logger = logging.getLogger("dashboard")

class Dashboard:
    """
    Interactive dashboard for visualizing market data, predictions and trading performance
    """
    
    def __init__(self, trading_system: TradingSystem, refresh_interval: int = 30):
        """
        Initialize the dashboard
        
        Args:
            trading_system (TradingSystem): Trading system instance
            refresh_interval (int): Data refresh interval in seconds
        """
        self.trading_system = trading_system
        self.refresh_interval = refresh_interval
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        self.last_update = datetime.now()
        self.current_data = {}
        self.symbol_data = {}
        self.portfolio_history = []
        self.running = False
        self.update_thread = None
        
        # Add health monitoring endpoints
        self._setup_monitoring()
        
        # Configure layout
        self._setup_layout()
        # Configure callbacks
        self._setup_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _setup_monitoring(self):
        """Setup monitoring endpoints for health checks"""
        server = self.app.server
        
        @server.route("/health")
        def health():
            """Simple health check endpoint"""
            return jsonify(dashboard_monitor.get_simple_health())
        
        @server.route("/health/detailed")
        def health_detailed():
            """Detailed health check endpoint"""
            return jsonify(dashboard_monitor.get_health_status())
            
        logger.info("Health monitoring endpoints configured at /health and /health/detailed")
    
    def _setup_layout(self):
        """Configure dashboard layout"""
        self.app.layout = html.Div([
            # Header
            dbc.Navbar(
                dbc.Container([
                    html.A(
                        dbc.Row([
                            dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                            dbc.Col(dbc.NavbarBrand("CryptoVisionAI", className="ml-2"), width="auto")
                        ], align="center"),
                        href="/",
                    ),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                            dbc.NavItem(dbc.NavLink("Trading", href="#")),
                            dbc.NavItem(dbc.NavLink("Models", href="#")),
                            dbc.NavItem(dbc.NavLink("Settings", href="#")),
                            dbc.NavItem(dbc.Button("Start Trading", color="success", id="trading-button")),
                            dbc.NavItem(
                                dbc.Button(
                                    html.I(className="fas fa-heartbeat"), 
                                    color="info", 
                                    className="ml-2", 
                                    id="health-button", 
                                    outline=True,
                                    size="sm"
                                )
                            ),
                        ], className="ml-auto", navbar=True),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ], fluid=True),
                color="dark",
                dark=True,
            ),
            
            # Health monitor modal
            dbc.Modal([
                dbc.ModalHeader("System Health Monitor"),
                dbc.ModalBody([
                    dbc.Spinner(html.Div(id="health-monitor-content")),
                    dcc.Interval(
                        id="health-update-interval",
                        interval=5000,  # update every 5 seconds when open
                        n_intervals=0,
                        disabled=True
                    )
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-health-modal", className="ml-auto")
                ),
            ], id="health-monitor-modal", size="lg"),
            
            # Main content
            dbc.Container([
                html.Div(id="system-alerts"),
                
                # Status and controls row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("System Status"),
                            dbc.CardBody([
                                html.Div([
                                    html.Span("Status: "),
                                    html.Span(id="system-status", children="Stopped")
                                ]),
                                html.Div([
                                    html.Span("Last Update: "),
                                    html.Span(id="last-update-time")
                                ]),
                                html.Div([
                                    html.Span("API Latency: "),
                                    html.Span(id="api-latency")
                                ]),
                                html.Div([
                                    html.Span("Models Loaded: "),
                                    html.Span(id="models-loaded")
                                ])
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardHeader("Market Selection"),
                            dbc.CardBody([
                                dcc.Dropdown(
                                    id="symbol-dropdown",
                                    options=[],
                                    value="BTCUSDT",
                                    clearable=False,
                                    className="mb-3"
                                ),
                                dcc.Dropdown(
                                    id="timeframe-dropdown",
                                    options=[
                                        {"label": "1 minute", "value": "1m"},
                                        {"label": "5 minutes", "value": "5m"},
                                        {"label": "15 minutes", "value": "15m"},
                                        {"label": "1 hour", "value": "1h"},
                                        {"label": "4 hours", "value": "4h"},
                                        {"label": "1 day", "value": "1d"}
                                    ],
                                    value="1h",
                                    clearable=False,
                                    className="mb-3"
                                ),
                                dcc.RadioItems(
                                    id="chart-type",
                                    options=[
                                        {"label": "Candlestick", "value": "candlestick"},
                                        {"label": "Line", "value": "line"},
                                        {"label": "OHLC", "value": "ohlc"}
                                    ],
                                    value="candlestick",
                                    inline=True
                                )
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Portfolio Summary"),
                            dbc.CardBody([
                                html.Div([
                                    html.H4(id="portfolio-value", className="m-0"),
                                    html.Div(id="portfolio-change", className="mt-1")
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.P("Initial Capital", className="text-muted mb-0"),
                                        html.H5(id="initial-capital", className="mt-0")
                                    ]),
                                    dbc.Col([
                                        html.P("Profit/Loss", className="text-muted mb-0"),
                                        html.H5(id="profit-loss", className="mt-0")
                                    ]),
                                    dbc.Col([
                                        html.P("Win Rate", className="text-muted mb-0"),
                                        html.H5(id="win-rate", className="mt-0")
                                    ])
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.P("Open Positions", className="text-muted mb-0"),
                                        html.H5(id="open-positions-count", className="mt-0")
                                    ]),
                                    dbc.Col([
                                        html.P("Closed Positions", className="text-muted mb-0"),
                                        html.H5(id="closed-positions-count", className="mt-0")
                                    ]),
                                    dbc.Col([
                                        html.P("Current Exposure", className="text-muted mb-0"),
                                        html.H5(id="current-exposure", className="mt-0")
                                    ])
                                ])
                            ])
                        ], className="mb-4"),
                        
                        dbc.Card([
                            dbc.CardHeader("Latest Prediction"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H3(id="prediction-symbol", className="m-0"),
                                        html.H5(id="prediction-price", className="mt-1")
                                    ], width=4),
                                    dbc.Col([
                                        html.Div(id="prediction-signal"),
                                        html.Div([
                                            html.Span("Probability: "),
                                            html.Span(id="prediction-probability")
                                        ]),
                                        html.Div([
                                            html.Span("Signal Strength: "),
                                            html.Span(id="prediction-strength")
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        html.P("RSI", className="text-muted mb-0"),
                                        html.H5(id="prediction-rsi", className="mt-0"),
                                        html.P("Volatility", className="text-muted mb-0"),
                                        html.H5(id="prediction-volatility", className="mt-0")
                                    ], width=4)
                                ]),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Button(
                                            "Execute Buy", 
                                            id="execute-buy-button",
                                            className="btn btn-success mr-2"
                                        ),
                                        html.Button(
                                            "Execute Sell", 
                                            id="execute-sell-button",
                                            className="btn btn-danger ml-2"
                                        )
                                    ], className="text-center")
                                ])
                            ])
                        ])
                    ], width=9)
                ], className="mb-4"),
                
                # Charts row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Price Chart"),
                            dbc.CardBody([
                                dcc.Graph(id="price-chart", style={"height": "400px"})
                            ])
                        ])
                    ], width=8),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Indicators"),
                            dbc.CardBody([
                                dcc.Graph(id="indicators-chart", style={"height": "400px"})
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                # Performance and positions row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Performance Metrics"),
                            dbc.CardBody([
                                dcc.Graph(id="performance-chart", style={"height": "300px"})
                            ])
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                "Open Positions",
                                dbc.Button(
                                    "Refresh", 
                                    id="refresh-positions-button",
                                    color="primary",
                                    size="sm",
                                    className="float-right"
                                )
                            ]),
                            dbc.CardBody([
                                html.Div(id="open-positions-table", style={"overflowY": "auto", "maxHeight": "300px"})
                            ])
                        ])
                    ], width=6)
                ], className="mb-4"),
                
                # Trade history row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                "Trade History",
                                dbc.Button(
                                    "Download CSV", 
                                    id="download-history-button",
                                    color="primary",
                                    size="sm",
                                    className="float-right"
                                )
                            ]),
                            dbc.CardBody([
                                html.Div(id="trade-history-table", style={"overflowY": "auto", "maxHeight": "300px"})
                            ])
                        ])
                    ], width=12)
                ]),
                
                # Hidden storage
                dcc.Store(id="portfolio-data-store"),
                dcc.Store(id="market-data-store"),
                dcc.Store(id="predictions-store"),
                dcc.Store(id="health-data-store"),
                
                # Interval for updates
                dcc.Interval(
                    id="interval-component",
                    interval=self.refresh_interval * 1000,  # in milliseconds
                    n_intervals=0
                )
            ], fluid=True, className="mt-4")
        ])
    
    def _setup_callbacks(self):
        """Configure dashboard callbacks"""
        
        # Health monitoring callbacks
        @self.app.callback(
            Output("health-monitor-modal", "is_open"),
            [
                Input("health-button", "n_clicks"),
                Input("close-health-modal", "n_clicks")
            ],
            [State("health-monitor-modal", "is_open")],
        )
        def toggle_health_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open
        
        @self.app.callback(
            Output("health-update-interval", "disabled"),
            [Input("health-monitor-modal", "is_open")]
        )
        def toggle_health_interval(is_open):
            return not is_open
        
        @self.app.callback(
            [
                Output("health-monitor-content", "children"),
                Output("health-data-store", "data")
            ],
            [Input("health-update-interval", "n_intervals")]
        )
        def update_health_monitor(n):
            health_data = dashboard_monitor.get_health_status()
            
            # Format health data for display
            system_info = health_data.get("system_info", {})
            memory_usage = health_data.get("memory_usage", {})
            cpu_usage = health_data.get("cpu_usage", {})
            disk_space = health_data.get("disk_space", {})
            uptime = health_data.get("uptime", {})
            
            status_color = {
                "healthy": "success",
                "warning": "warning",
                "error": "danger"
            }.get(health_data.get("status", ""), "warning")
            
            content = [
                dbc.Alert(
                    f"System Status: {health_data.get('status', 'Unknown').upper()}", 
                    color=status_color
                ),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("System Information"),
                            dbc.CardBody([
                                html.P(f"Platform: {system_info.get('platform')} {system_info.get('platform_release')}"),
                                html.P(f"Python Version: {system_info.get('python_version')}"),
                                html.P(f"Architecture: {system_info.get('architecture')}"),
                                html.P(f"Processor: {system_info.get('processor')}")
                            ])
                        ]),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Uptime Information"),
                            dbc.CardBody([
                                html.P(f"Start Time: {uptime.get('start_time')}"),
                                html.P(f"Uptime: {uptime.get('uptime')}"),
                                html.P(f"Current Time: {health_data.get('timestamp')}")
                            ])
                        ]),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Memory Usage"),
                            dbc.CardBody([
                                dbc.Progress(
                                    value=memory_usage.get("system_memory_used_percent", 0),
                                    color="warning" if memory_usage.get("system_memory_used_percent", 0) > 70 else "success",
                                    className="mb-3",
                                    label=f"{memory_usage.get('system_memory_used_percent', 0):.1f}%"
                                ),
                                html.P(f"Process Memory: {memory_usage.get('process_memory_rss_mb', 0):.2f} MB"),
                                html.P(f"System Memory: {memory_usage.get('system_memory_available_gb', 0):.2f} GB available of {memory_usage.get('system_memory_total_gb', 0):.2f} GB")
                            ])
                        ]),
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("CPU Usage"),
                            dbc.CardBody([
                                dbc.Progress(
                                    value=cpu_usage.get("system_cpu_percent", 0),
                                    color="warning" if cpu_usage.get("system_cpu_percent", 0) > 70 else "success",
                                    className="mb-3",
                                    label=f"{cpu_usage.get('system_cpu_percent', 0):.1f}%"
                                ),
                                html.P(f"Process CPU: {cpu_usage.get('process_cpu_percent', 0):.2f}%"),
                                html.P(f"System CPU: {cpu_usage.get('system_cpu_percent', 0):.2f}%"),
                                html.P(f"CPU Count: {cpu_usage.get('cpu_count', 0)} cores")
                            ])
                        ]),
                    ], width=6),
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Disk Space"),
                            dbc.CardBody([
                                dbc.Progress(
                                    value=disk_space.get("disk_used_percent", 0),
                                    color="warning" if disk_space.get("disk_used_percent", 0) > 70 else "success",
                                    className="mb-3",
                                    label=f"{disk_space.get('disk_used_percent', 0):.1f}%"
                                ),
                                html.P(f"Free: {disk_space.get('disk_free_gb', 0):.2f} GB"),
                                html.P(f"Used: {disk_space.get('disk_used_gb', 0):.2f} GB of {disk_space.get('disk_total_gb', 0):.2f} GB")
                            ])
                        ]),
                    ], width=12),
                ]),
                
                # Show warnings if any
                dbc.Alert(
                    [html.P("Warnings:"), html.Ul([html.Li(w) for w in health_data.get("warnings", [])])],
                    color="warning",
                    className="mt-3"
                ) if health_data.get("warnings") else None,
            ]
            
            return content, health_data
        
        # Symbol dropdown options
        @self.app.callback(
            Output("symbol-dropdown", "options"),
            Input("interval-component", "n_intervals")
        )
        def update_symbol_options(n):
            active_symbols = self.trading_system.get_active_symbols()
            options = [{"label": symbol, "value": symbol} for symbol in active_symbols]
            return options
            
        # Update system status
        @self.app.callback(
            [
                Output("system-status", "children"),
                Output("system-status", "className"),
                Output("last-update-time", "children"),
                Output("api-latency", "children"),
                Output("models-loaded", "children"),
                Output("trading-button", "children"),
                Output("trading-button", "color")
            ],
            Input("interval-component", "n_intervals")
        )
        def update_system_status(n):
            system_status = self.trading_system.get_system_status()
            status_text = system_status.get("status", "Unknown")
            status_class = "text-success" if status_text == "running" else "text-danger"
            last_update = system_status.get("last_update", datetime.now().isoformat())
            api_latency = system_status.get("api_latency", "N/A")
            if api_latency:
                api_latency = f"{api_latency:.2f} ms"
            models_loaded = "Yes" if system_status.get("model_loaded", False) else "No"
            button_text = "Stop Trading" if status_text == "running" else "Start Trading"
            button_color = "danger" if status_text == "running" else "success"
            
            return status_text, status_class, last_update, api_latency, models_loaded, button_text, button_color
            
        # Update portfolio summary
        @self.app.callback(
            [
                Output("portfolio-value", "children"),
                Output("portfolio-change", "children"),
                Output("portfolio-change", "className"),
                Output("initial-capital", "children"),
                Output("profit-loss", "children"),
                Output("profit-loss", "className"),
                Output("win-rate", "children"),
                Output("open-positions-count", "children"),
                Output("closed-positions-count", "children"),
                Output("current-exposure", "children"),
                Output("portfolio-data-store", "data")
            ],
            Input("interval-component", "n_intervals")
        )
        def update_portfolio_summary(n):
            portfolio_summary = self.trading_system.get_portfolio_summary()
            
            # Format portfolio value
            portfolio_value = portfolio_summary.get("portfolio_value", 0)
            formatted_value = f"${portfolio_value:,.2f}"
            
            # Calculate and format change
            profit_loss = portfolio_summary.get("profit_loss", 0)
            profit_loss_percent = portfolio_summary.get("profit_loss_percent", 0)
            change_text = f"{profit_loss:+,.2f} ({profit_loss_percent:+.2f}%)"
            change_class = "text-success" if profit_loss >= 0 else "text-danger"
            
            # Format initial capital
            initial_capital = portfolio_summary.get("initial_capital", 0)
            formatted_initial = f"${initial_capital:,.2f}"
            
            # Format profit/loss
            formatted_pnl = f"${profit_loss:+,.2f}"
            pnl_class = "text-success" if profit_loss >= 0 else "text-danger"
            
            # Win rate
            metrics = portfolio_summary.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            formatted_win_rate = f"{win_rate:.1f}%"
            
            # Position counts
            open_positions = portfolio_summary.get("open_positions", 0)
            closed_positions = portfolio_summary.get("closed_positions", 0)
            
            # Current exposure
            exposure = portfolio_summary.get("current_exposure", 0)
            exposure_percent = portfolio_summary.get("exposure_percent", 0)
            formatted_exposure = f"${exposure:,.2f} ({exposure_percent:.1f}%)"
            
            # Store full portfolio data for other callbacks
            portfolio_data = portfolio_summary
            
            # Add to portfolio history for the chart
            self.portfolio_history.append({
                "timestamp": datetime.now().isoformat(),
                "value": portfolio_value
            })
            # Keep last 100 points
            if len(self.portfolio_history) > 100:
                self.portfolio_history = self.portfolio_history[-100:]
                
            return (
                formatted_value, 
                change_text,
                change_class,
                formatted_initial,
                formatted_pnl,
                pnl_class,
                formatted_win_rate,
                open_positions,
                closed_positions,
                formatted_exposure,
                portfolio_data
            )
        
        # Update market data and chart
        @self.app.callback(
            [
                Output("price-chart", "figure"),
                Output("indicators-chart", "figure"),
                Output("market-data-store", "data")
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("symbol-dropdown", "value"),
                Input("timeframe-dropdown", "value"),
                Input("chart-type", "value")
            ]
        )
        def update_charts(n, symbol, timeframe, chart_type):
            if not symbol:
                symbol = "BTCUSDT"
                
            # Get historical data
            df = self.trading_system._get_historical_data(symbol, timeframe)
            
            if df is None or len(df) < 10:
                return go.Figure(), go.Figure(), {}
            
            # Store for other callbacks
            market_data = df.reset_index().to_dict("records")
            
            # Create price chart
            fig_price = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Add price data based on chart type
            if chart_type == "candlestick":
                fig_price.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
            elif chart_type == "line":
                fig_price.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['close'],
                        name="Price",
                        line=dict(color="#2962FF")
                    ),
                    row=1, col=1
                )
            elif chart_type == "ohlc":
                fig_price.add_trace(
                    go.Ohlc(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
            
            # Add volume bars
            colors = ['#2962FF' if row['open'] <= row['close'] else '#FF2929' for _, row in df.iterrows()]
            fig_price.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    marker_color=colors,
                    name="Volume"
                ),
                row=2, col=1
            )
            
            # Add moving averages
            ma50 = df['close'].rolling(window=50).mean()
            ma200 = df['close'].rolling(window=200).mean()
            
            fig_price.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma50,
                    name="MA(50)",
                    line=dict(color="orange", width=1)
                ),
                row=1, col=1
            )
            
            fig_price.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma200,
                    name="MA(200)",
                    line=dict(color="purple", width=1)
                ),
                row=1, col=1
            )
            
            # Update layout
            fig_price.update_layout(
                title=f"{symbol} - {timeframe}",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_rangeslider_visible=False
            )
            
            # Create indicators chart
            fig_indicators = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.5]
            )
            
            # Calculate RSI
            delta = df['close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = 100 - (100 / (1 + rs))
            
            # Add RSI
            fig_indicators.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi,
                    name="RSI(14)",
                    line=dict(color="#17BECF")
                ),
                row=1, col=1
            )
            
            # Add RSI reference lines
            fig_indicators.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=1, col=1)
            fig_indicators.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=1, col=1)
            
            # Calculate MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # Add MACD
            fig_indicators.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd,
                    name="MACD",
                    line=dict(color="#17BECF")
                ),
                row=2, col=1
            )
            
            fig_indicators.add_trace(
                go.Scatter(
                    x=df.index,
                    y=signal,
                    name="Signal",
                    line=dict(color="#FF9900")
                ),
                row=2, col=1
            )
            
            histogram_colors = ['#2962FF' if val >= 0 else '#FF2929' for val in histogram]
            fig_indicators.add_trace(
                go.Bar(
                    x=df.index,
                    y=histogram,
                    name="Histogram",
                    marker_color=histogram_colors
                ),
                row=2, col=1
            )
            
            # Update layout
            fig_indicators.update_layout(
                title="Technical Indicators",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="RSI",
                yaxis2_title="MACD"
            )
            
            return fig_price, fig_indicators, market_data
            
        # Update prediction
        @self.app.callback(
            [
                Output("prediction-symbol", "children"),
                Output("prediction-price", "children"),
                Output("prediction-signal", "children"),
                Output("prediction-signal", "className"),
                Output("prediction-probability", "children"),
                Output("prediction-strength", "children"),
                Output("prediction-rsi", "children"),
                Output("prediction-volatility", "children"),
                Output("predictions-store", "data")
            ],
            [
                Input("interval-component", "n_intervals"),
                Input("symbol-dropdown", "value")
            ]
        )
        def update_prediction(n, symbol):
            if not symbol:
                symbol = "BTCUSDT"
                
            # Get prediction
            prediction = self.trading_system.get_prediction(symbol)
            
            if prediction is None or "error" in prediction:
                error_msg = prediction.get("error", "Failed to get prediction") if prediction else "No prediction available"
                return (
                    symbol,
                    "N/A",
                    "No Signal",
                    "",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    {}
                )
            
            # Format prediction data
            current_price = prediction.get("current_price", 0)
            formatted_price = f"${current_price:,.2f}"
            
            pred_data = prediction.get("prediction", {})
            signal = pred_data.get("signal", "hold").upper()
            probability = pred_data.get("probability", 0.5)
            strength = pred_data.get("strength", 0)
            
            # Set signal class
            if signal == "BUY":
                signal_class = "alert alert-success p-1"
            elif signal == "SELL":
                signal_class = "alert alert-danger p-1"
            else:
                signal_class = "alert alert-warning p-1"
                
            # Format additional data
            rsi = prediction.get("rsi")
            formatted_rsi = f"{rsi:.1f}" if rsi is not None else "N/A"
            
            volatility = prediction.get("volatility")
            formatted_vol = f"{volatility:.2%}" if volatility is not None else "N/A"
            
            # Format probability and strength
            formatted_prob = f"{probability:.1%}"
            formatted_strength = f"{strength:.1%}"
            
            return (
                symbol,
                formatted_price,
                signal,
                signal_class,
                formatted_prob,
                formatted_strength,
                formatted_rsi,
                formatted_vol,
                prediction
            )
            
        # Update performance chart
        @self.app.callback(
            Output("performance-chart", "figure"),
            Input("portfolio-data-store", "data")
        )
        def update_performance_chart(portfolio_data):
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "indicator"}, {"type": "pie"}]],
                column_widths=[0.4, 0.6],
                subplot_titles=["Sharpe Ratio", "Win/Loss Ratio"]
            )
            
            # Add Sharpe indicator
            metrics = portfolio_data.get("metrics", {}) if portfolio_data else {}
            sharpe = metrics.get("sharpe_ratio", 0)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sharpe,
                    number={"valueformat": ".2f"},
                    title={"text": "Sharpe Ratio"},
                    gauge={
                        "axis": {"range": [-1, 3]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [-1, 0], "color": "red"},
                            {"range": [0, 1], "color": "orange"},
                            {"range": [1, 2], "color": "yellow"},
                            {"range": [2, 3], "color": "green"}
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.75,
                            "value": 1
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Add Win/Loss pie chart
            win_count = metrics.get("win_count", 0)
            loss_count = metrics.get("loss_count", 0)
            
            if win_count > 0 or loss_count > 0:
                fig.add_trace(
                    go.Pie(
                        labels=["Wins", "Losses"],
                        values=[win_count, loss_count],
                        textinfo="label+percent",
                        marker=dict(colors=['#00CC96', '#EF553B'])
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            return fig
            
        # Update open positions table
        @self.app.callback(
            Output("open-positions-table", "children"),
            [
                Input("interval-component", "n_intervals"),
                Input("refresh-positions-button", "n_clicks")
            ]
        )
        def update_open_positions_table(n, n_clicks):
            positions = self.trading_system.portfolio.get_all_positions() if self.trading_system.portfolio else {}
            open_positions = positions.get("open_positions", [])
            
            if not open_positions:
                return html.Div("No open positions", className="text-center p-3 text-muted")
            
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Type"),
                    html.Th("Entry Price"),
                    html.Th("Amount"),
                    html.Th("Current PnL"),
                    html.Th("Stop Loss"),
                    html.Th("Take Profit"),
                    html.Th("Actions")
                ]))
            ]
            
            rows = []
            for pos in open_positions:
                # Add row for each position
                row = html.Tr([
                    html.Td(pos["symbol"]),
                    html.Td(html.Span(pos["position_type"].capitalize(), className="badge badge-primary")),
                    html.Td(f"${pos['entry_price']:,.2f}"),
                    html.Td(f"{pos['amount']:.6f}"),
                    html.Td(
                        f"${pos['pnl']:+,.2f} ({pos['pnl_percent']:+.2f}%)",
                        className="text-success" if pos['pnl'] >= 0 else "text-danger"
                    ),
                    html.Td(f"${pos['stop_loss']:,.2f}" if pos['stop_loss'] else "-"),
                    html.Td(f"${pos['take_profit']:,.2f}" if pos['take_profit'] else "-"),
                    html.Td(
                        html.Button(
                            "Close", 
                            id={"type": "close-position-button", "index": pos["position_id"]},
                            className="btn btn-danger btn-sm",
                            n_clicks=0
                        )
                    )
                ])
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True, size="sm")
            
            return table
            
        # Update trade history table
        @self.app.callback(
            Output("trade-history-table", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_trade_history_table(n):
            positions = self.trading_system.portfolio.get_all_positions() if self.trading_system.portfolio else {}
            closed_positions = positions.get("closed_positions", [])
            
            if not closed_positions:
                return html.Div("No trade history", className="text-center p-3 text-muted")
            
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Type"),
                    html.Th("Entry Price"),
                    html.Th("Exit Price"),
                    html.Th("Amount"),
                    html.Th("PnL"),
                    html.Th("Entry Time"),
                    html.Th("Exit Time"),
                    html.Th("Duration")
                ]))
            ]
            
            rows = []
            for pos in closed_positions:
                # Add row for each position
                row = html.Tr([
                    html.Td(pos["symbol"]),
                    html.Td(html.Span(pos["position_type"].capitalize(), className="badge badge-primary")),
                    html.Td(f"${pos['entry_price']:,.2f}"),
                    html.Td(f"${pos['exit_price']:,.2f}"),
                    html.Td(f"{pos['amount']:.6f}"),
                    html.Td(
                        f"${pos['pnl']:+,.2f} ({pos['pnl_percent']:+.2f}%)",
                        className="text-success" if pos['pnl'] >= 0 else "text-danger"
                    ),
                    html.Td(datetime.fromisoformat(pos["entry_time"]).strftime("%Y-%m-%d %H:%M:%S") if pos['entry_time'] else "-"),
                    html.Td(datetime.fromisoformat(pos["exit_time"]).strftime("%Y-%m-%d %H:%M:%S") if pos['exit_time'] else "-"),
                    html.Td(pos["duration"] or "-")
                ])
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True, size="sm")
            
            return table
            
        # Start/stop trading button callback
        @self.app.callback(
            Output("system-alerts", "children"),
            Input("trading-button", "n_clicks"),
            State("system-status", "children")
        )
        def toggle_trading(n_clicks, current_status):
            if n_clicks is None:
                return no_update
            
            if current_status == "running":
                self.trading_system.stop()
                return dbc.Alert("Trading stopped", color="warning", dismissable=True, duration=5000)
            else:
                self.trading_system.start()
                return dbc.Alert("Trading started", color="success", dismissable=True, duration=5000)
                
        # Execute buy button callback
        @self.app.callback(
            Output("system-alerts", "children", allow_duplicate=True),
            [
                Input("execute-buy-button", "n_clicks")
            ],
            [
                State("symbol-dropdown", "value"),
                State("prediction-price", "children")
            ],
            prevent_initial_call=True
        )
        def execute_buy(n_clicks, symbol, current_price):
            if n_clicks is None or not symbol:
                return no_update
            
            # Extract current price and calculate amount
            try:
                # Default amount
                amount = 0.001  # Small BTC amount for example
                
                result = self.trading_system.execute_trade(symbol, "buy", amount)
                
                if "error" in result:
                    return dbc.Alert(f"Error: {result['error']}", color="danger", dismissable=True)
                
                return dbc.Alert(
                    f"Buy order executed: {amount} {symbol} at {current_price}", 
                    color="success", 
                    dismissable=True
                )
            except Exception as e:
                return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
                
        # Execute sell button callback
        @self.app.callback(
            Output("system-alerts", "children", allow_duplicate=True),
            [
                Input("execute-sell-button", "n_clicks")
            ],
            [
                State("symbol-dropdown", "value"),
                State("prediction-price", "children")
            ],
            prevent_initial_call=True
        )
        def execute_sell(n_clicks, symbol, current_price):
            if n_clicks is None or not symbol:
                return no_update
            
            # Extract current price and calculate amount
            try:
                # Default amount
                amount = 0.001  # Small BTC amount for example
                
                result = self.trading_system.execute_trade(symbol, "sell", amount)
                
                if "error" in result:
                    return dbc.Alert(f"Error: {result['error']}", color="danger", dismissable=True)
                
                return dbc.Alert(
                    f"Sell order executed: {amount} {symbol} at {current_price}", 
                    color="success", 
                    dismissable=True
                )
            except Exception as e:
                return dbc.Alert(f"Error: {str(e)}", color="danger", dismissable=True)
    
    def start(self, debug: bool = False, host: str = "0.0.0.0", port: int = 8050) -> None:
        """
        Start the dashboard server
        
        Args:
            debug (bool): Whether to run in debug mode
            host (str): Host to run the server on
            port (int): Port to run the server on
        """
        logger.info(f"Starting dashboard on http://{host}:{port}/")
        logger.info(f"Health monitoring available at http://{host}:{port}/health")
        self.running = True
        self.app.run(debug=debug, host=host, port=port)
    
    def stop(self) -> None:
        """Stop the dashboard"""
        self.running = False
        logger.info("Dashboard stopped")


# Example usage
if __name__ == "__main__":
    from src.trading.trading_system import TradingSystem
    
    # Create trading system
    config = {
        "use_testnet": True,
        "test_mode": True,
        "initial_capital": 10000.0
    }
    trading_system = TradingSystem(config)
    
    # Create and start dashboard
    dashboard = Dashboard(trading_system)
    dashboard.start(debug=True)