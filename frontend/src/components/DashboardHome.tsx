import React, { useState } from 'react';
import { Container } from '@mui/material';
import { Responsive, WidthProvider } from 'react-grid-layout/legacy';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

import {
    Edit as EditIcon,
    Save as SaveIcon,
} from '@mui/icons-material';

import PortfolioSummary from './PortfolioSummary';
import MarketDataChart from './MarketDataChart';
import OrderBook from './OrderBook';
import RecentTrades from './RecentTrades';
import OrderForm from './OrderForm';
import OrderManagement from './OrderManagement';
import RiskCalculator from './RiskCalculator';
import PerformanceMetrics from './PerformanceMetrics';
import PortfolioAllocation from './PortfolioAllocation';
import RiskMonitor from './RiskMonitor';
import RiskSettings from './RiskSettings';
import RiskAlerts from './RiskAlerts';
import CorrelationMatrix from './CorrelationMatrix';
import VolatilityIndicator from './VolatilityIndicator';
import KeyboardShortcuts from './KeyboardShortcuts';
import Watchlist from './Watchlist';
import Settings from './Settings';
import SentimentAnalysis from './SentimentAnalysis';
import AIPrediction from './AIPrediction';
import NewsFeed from './NewsFeed';
import StrategyBuilder from './StrategyBuilder';
import ModelPerformanceCard from './ModelPerformanceCard';
import { Button, Box } from '@mui/material';

const ResponsiveGridLayout = WidthProvider(Responsive);

// Initial layout configuration - Optimized for clarity & width
const initialLayouts = {
    lg: [
        // Top Row: Critical Summary (Full Width for wider chips)
        { i: 'portfolio_summary', x: 0, y: 0, w: 12, h: 4 },

        // Main Stage: Risk Monitor (Top Right context) + Market Chart
        // Move Risk Monitor to side of chart? Or keep chart big?
        // Let's put Risk Monitor with Watchlist
        { i: 'watchlist', x: 0, y: 4, w: 2, h: 10 },
        { i: 'market_chart', x: 2, y: 4, w: 7, h: 10 },
        { i: 'risk_monitor', x: 9, y: 4, w: 3, h: 10 }, // Vertical Risk Monitor?

        // Trading Action Row
        { i: 'order_form', x: 0, y: 14, w: 4, h: 8 },
        { i: 'order_book', x: 4, y: 14, w: 4, h: 8 },
        { i: 'recent_trades', x: 8, y: 14, w: 4, h: 8 },

        // Secondary Info / Order Management
        { i: 'order_management', x: 0, y: 22, w: 8, h: 6 },
        { i: 'volatility', x: 8, y: 22, w: 4, h: 6 },

        // Analytics Row 1
        { i: 'performance', x: 0, y: 28, w: 6, h: 6 },
        { i: 'allocation', x: 6, y: 28, w: 3, h: 6 },
        { i: 'model_performance', x: 9, y: 28, w: 3, h: 6 },

        // Analytics Row 2
        { i: 'prediction', x: 0, y: 34, w: 4, h: 6 },
        { i: 'sentiment', x: 4, y: 34, w: 4, h: 6 },
        { i: 'news', x: 8, y: 34, w: 4, h: 6 },

        // Deep Dive Analysis (Bottom)
        { i: 'correlation', x: 0, y: 40, w: 6, h: 6 },
        { i: 'risk_settings', x: 6, y: 40, w: 6, h: 6 },

        // Strategy & Settings (Full Width)
        { i: 'strategy', x: 0, y: 46, w: 12, h: 8 },
        { i: 'settings_page', x: 0, y: 54, w: 12, h: 8 },
    ],
};

interface DashboardHomeProps {
    isDraggable: boolean;
    setIsDraggable: (value: boolean) => void;
}

const DashboardHome: React.FC<DashboardHomeProps> = ({ isDraggable, setIsDraggable }) => {
    const [selectedSymbol, setSelectedSymbol] = useState<string>('BTCUSDT');
    const [currentPrice] = useState(0);
    const [accountBalance] = useState(10000);
    const [riskCalcOpen, setRiskCalcOpen] = useState(false);

    // Layout state
    const [layouts, setLayouts] = useState(() => {
        const saved = localStorage.getItem('dashboard_layouts');
        return saved ? JSON.parse(saved) : initialLayouts;
    });

    const handleLayoutChange = (_layout: any, allLayouts: any) => {
        setLayouts(allLayouts);
        localStorage.setItem('dashboard_layouts', JSON.stringify(allLayouts));
    };

    return (
        <React.Fragment>
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                <Button
                    onClick={() => {
                        setLayouts(initialLayouts);
                        localStorage.removeItem('dashboard_layouts');
                    }}
                    color="secondary"
                    variant="outlined"
                >
                    Reset Layout
                </Button>
                <Button
                    startIcon={isDraggable ? <SaveIcon /> : <EditIcon />}
                    onClick={() => setIsDraggable(!isDraggable)}
                    color="primary"
                    variant="outlined"
                >
                    {isDraggable ? 'Save Layout' : 'Edit Layout'}
                </Button>
            </Box>

            <RiskAlerts />
            <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                <ResponsiveGridLayout
                    className="layout"
                    layouts={layouts}
                    breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                    cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
                    rowHeight={30}
                    isDraggable={isDraggable}
                    isResizable={isDraggable}
                    onLayoutChange={handleLayoutChange}
                    draggableHandle=".drag-handle"
                    compactType="vertical"
                    preventCollision={false}
                >
                    <div key="risk_monitor" className={isDraggable ? 'drag-handle' : ''}>
                        <RiskMonitor />
                    </div>
                    <div key="portfolio_summary" className={isDraggable ? 'drag-handle' : ''}>
                        <PortfolioSummary />
                    </div>
                    <div key="volatility" className={isDraggable ? 'drag-handle' : ''}>
                        <VolatilityIndicator symbol={selectedSymbol} />
                    </div>
                    <div key="market_chart" className={isDraggable ? 'drag-handle' : ''}>
                        <MarketDataChart />
                    </div>
                    <div key="watchlist" className={isDraggable ? 'drag-handle' : ''}>
                        <Watchlist onSelectSymbol={setSelectedSymbol} />
                    </div>
                    <div key="order_form" className={isDraggable ? 'drag-handle' : ''}>
                        <OrderForm symbol={selectedSymbol} currentPrice={currentPrice || 50000} />
                    </div>
                    <div key="order_book" className={isDraggable ? 'drag-handle' : ''}>
                        <OrderBook symbol={selectedSymbol} />
                    </div>
                    <div key="recent_trades" className={isDraggable ? 'drag-handle' : ''}>
                        <RecentTrades symbol={selectedSymbol} />
                    </div>
                    <div key="order_management" className={isDraggable ? 'drag-handle' : ''}>
                        <OrderManagement symbol={selectedSymbol} />
                    </div>
                    <div key="performance" className={isDraggable ? 'drag-handle' : ''}>
                        <PerformanceMetrics />
                    </div>
                    <div key="allocation" className={isDraggable ? 'drag-handle' : ''}>
                        <PortfolioAllocation />
                    </div>
                    <div key="model_performance" className={isDraggable ? 'drag-handle' : ''}>
                        <ModelPerformanceCard />
                    </div>
                    <div key="correlation" className={isDraggable ? 'drag-handle' : ''}>
                        <CorrelationMatrix />
                    </div>
                    <div key="risk_settings" className={isDraggable ? 'drag-handle' : ''}>
                        <RiskSettings />
                    </div>
                    <div key="sentiment" className={isDraggable ? 'drag-handle' : ''}>
                        <SentimentAnalysis symbol={selectedSymbol} />
                    </div>
                    <div key="prediction" className={isDraggable ? 'drag-handle' : ''}>
                        <AIPrediction symbol={selectedSymbol} />
                    </div>
                    <div key="news" className={isDraggable ? 'drag-handle' : ''}>
                        <NewsFeed />
                    </div>
                    <div key="strategy" className={isDraggable ? 'drag-handle' : ''}>
                        <StrategyBuilder />
                    </div>
                    <div key="settings_page" className={isDraggable ? 'drag-handle' : ''}>
                        <Settings />
                    </div>
                </ResponsiveGridLayout>
                <KeyboardShortcuts
                    onFocusBuy={() => console.log('Focus Buy')}
                    onFocusSell={() => console.log('Focus Sell')}
                />
            </Container>

            <RiskCalculator
                open={riskCalcOpen}
                onClose={() => setRiskCalcOpen(false)}
                currentPrice={currentPrice || 50000}
                accountBalance={accountBalance}
                symbol={selectedSymbol}
            />
        </React.Fragment>
    );
};

export default DashboardHome;
