import { WS_BASE_URL, API_ENDPOINTS } from '../config/api';

export type MessageHandler = (data: any) => void;

export class WebSocketClient {
    private ws: WebSocket | null = null;
    private reconnectInterval = 5000;
    private reconnectTimer: number | null = null;
    private messageHandlers: Set<MessageHandler> = new Set();
    private symbol: string;
    private isIntentionallyClosed = false;

    constructor(symbol: string) {
        this.symbol = symbol;
    }

    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        this.isIntentionallyClosed = false;
        const wsUrl = `${WS_BASE_URL}${API_ENDPOINTS.wsMarket(this.symbol)}`;

        console.log(`Connecting to WebSocket: ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log(`WebSocket connected for ${this.symbol}`);
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.messageHandlers.forEach((handler) => handler(data));
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log(`WebSocket disconnected for ${this.symbol}`);
            if (!this.isIntentionallyClosed) {
                this.scheduleReconnect();
            }
        };
    }

    private scheduleReconnect() {
        if (this.reconnectTimer) {
            return;
        }

        console.log(`Reconnecting in ${this.reconnectInterval / 1000}s...`);
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, this.reconnectInterval);
    }

    onMessage(handler: MessageHandler) {
        this.messageHandlers.add(handler);
        return () => this.messageHandlers.delete(handler);
    }

    disconnect() {
        this.isIntentionallyClosed = true;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.messageHandlers.clear();
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}

export default WebSocketClient;
