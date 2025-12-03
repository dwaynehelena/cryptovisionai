import axios, { AxiosError } from 'axios';
import type { AxiosInstance } from 'axios';
import { API_BASE_URL, API_ENDPOINTS } from '../config/api';

class ApiService {
    private client: AxiosInstance;

    constructor() {
        this.client = axios.create({
            baseURL: API_BASE_URL,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Request interceptor
        this.client.interceptors.request.use(
            (config) => {
                console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
                return config;
            },
            (error) => Promise.reject(error)
        );

        // Response interceptor
        this.client.interceptors.response.use(
            (response) => response,
            (error: AxiosError) => {
                console.error('API Error:', error.response?.data || error.message);
                return Promise.reject(error);
            }
        );
    }

    // Health check
    async checkHealth() {
        const response = await this.client.get(API_ENDPOINTS.health);
        return response.data;
    }

    // Market data
    async getMarketData(symbol: string, interval: string = '1h', limit: number = 100) {
        const response = await this.client.get(API_ENDPOINTS.marketData, {
            params: { symbol, interval, limit },
        });
        return response.data;
    }

    async getMarketPrice(symbol: string) {
        const response = await this.client.get(API_ENDPOINTS.marketPrice(symbol));
        return response.data;
    }

    // Portfolio
    async getPortfolio() {
        const response = await this.client.get(API_ENDPOINTS.portfolio);
        return response.data;
    }

    async getPositions() {
        const response = await this.client.get(API_ENDPOINTS.positions);
        return response.data;
    }

    async getPosition(id: string) {
        const response = await this.client.get(API_ENDPOINTS.position(id));
        return response.data;
    }

    // Symbols
    async getSymbols() {
        const response = await this.client.get('/api/v1/market/symbols');
        return response.data;
    }

    // Order book
    async getOrderBook(symbol: string, limit: number = 20) {
        const response = await this.client.get(`/api/v1/market/orderbook/${symbol}`, {
            params: { limit },
        });
        return response.data;
    }

    // Recent trades
    async getRecentTrades(symbol: string, limit: number = 50) {
        const response = await this.client.get(`/api/v1/market/trades/${symbol}`, {
            params: { limit },
        });
        return response.data;
    }

    // Technical indicators
    async getIndicators(symbol: string, interval: string = '1h', limit: number = 100) {
        const response = await this.client.get(`/api/v1/market/indicators/${symbol}`, {
            params: { symbol, interval, limit },
        });
        return response.data;
    }

    // Orders
    async placeMarketOrder(order: any) {
        const response = await this.client.post('/api/v1/orders/market', order);
        return response.data;
    }

    async placeLimitOrder(order: any) {
        const response = await this.client.post('/api/v1/orders/limit', order);
        return response.data;
    }

    async placeStopOrder(order: any) {
        const response = await this.client.post('/api/v1/orders/stop', order);
        return response.data;
    }

    async getActiveOrders(symbol?: string) {
        const response = await this.client.get('/api/v1/orders/active', {
            params: symbol ? { symbol } : {},
        });
        return response.data;
    }

    async getOrderHistory(symbol: string, limit: number = 50) {
        const response = await this.client.get('/api/v1/orders/history', {
            params: { symbol, limit },
        });
        return response.data;
    }

    async cancelOrder(orderId: number, symbol: string) {
        const response = await this.client.delete(`/api/v1/orders/${orderId}`, {
            params: { symbol },
        });
        return response.data;
    }

    async calculateRisk(
        entryPrice: number,
        stopLoss: number,
        accountBalance: number,
        riskPercent: number = 2
    ) {
        const response = await this.client.get('/api/v1/orders/calculate-risk', {
            params: { entry_price: entryPrice, stop_loss: stopLoss, account_balance: accountBalance, risk_percent: riskPercent },
        });
        return response.data;
    }

    // Performance & Analytics
    async getPerformanceMetrics(timeframe: string = '1M') {
        const response = await this.client.get('/api/v1/portfolio/performance', {
            params: { timeframe },
        });
        return response.data;
    }

    async getPortfolioAllocation() {
        const response = await this.client.get('/api/v1/portfolio/allocation');
        return response.data;
    }

    async getTradeHistory(limit: number = 50) {
        const response = await this.client.get('/api/v1/portfolio/trades', {
            params: { limit },
        });
        return response.data;
    }

    // Risk Management
    async getRiskLimits() {
        const response = await this.client.get('/api/v1/risk/limits');
        return response.data;
    }

    async updateRiskLimits(limits: any) {
        const response = await this.client.post('/api/v1/risk/limits', limits);
        return response.data;
    }

    async getRiskStatus() {
        const response = await this.client.get('/api/v1/risk/status');
        return response.data;
    }

    async calculatePositionSize(entryPrice: number, stopPrice: number, volatility?: number) {
        const response = await this.client.post('/api/v1/risk/calculate-position-size', {
            entry_price: entryPrice,
            stop_price: stopPrice,
            volatility: volatility
        });
        return response.data;
    }

    async getRiskViolations(limit: number = 20) {
        const response = await this.client.get('/api/v1/risk/violations', {
            params: { limit }
        });
        return response.data;
    }

    async getCorrelationMatrix() {
        const response = await this.client.get('/api/v1/risk/correlation-matrix');
        return response.data;
    }

    async emergencyCloseAll() {
        const response = await this.client.post('/api/v1/risk/emergency-close-all', {
            confirmation: "CONFIRM_CLOSE_ALL"
        });
        return response.data;
    }

    async enableTrading() {
        const response = await this.client.post('/api/v1/risk/enable-trading');
        return response.data;
    }

    async disableTrading() {
        const response = await this.client.post('/api/v1/risk/disable-trading');
        return response.data;
    }
}

export const apiService = new ApiService();
export default apiService;
