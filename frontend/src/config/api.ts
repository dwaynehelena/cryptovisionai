// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
export const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000';

export const API_ENDPOINTS = {
    // Health
    health: '/health',

    // Market
    marketData: '/api/v1/market/data',
    marketPrice: (symbol: string) => `/api/v1/market/price/${symbol}`,

    // Portfolio
    portfolio: '/api/v1/portfolio/',
    positions: '/api/v1/portfolio/positions',
    position: (id: string) => `/api/v1/portfolio/positions/${id}`,

    // WebSocket
    wsMarket: (symbol: string) => `/ws/market/${symbol}`,
};
