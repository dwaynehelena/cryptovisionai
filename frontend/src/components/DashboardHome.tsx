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
import { Button, Box } from '@mui/material';

const ResponsiveGridLayout = WidthProvider(Responsive);

// Initial layout configuration
const initialLayouts = {
    lg: [
        { i: 'risk_monitor', x: 0, y: 0, w: 12, h: 4 },
        { i: 'portfolio_summary', x: 0, y: 4, w: 12, h: 3 },
        { i: 'volatility', x: 0, y: 7, w: 3, h: 4 },
        { i: 'market_chart', x: 3, y: 7, w: 9, h: 8 },
        { i: 'watchlist', x: 0, y: 11, w: 3, h: 6 },
        { i: 'order_form', x: 3, y: 15, w: 4, h: 6 },
        { i: 'order_book', x: 7, y: 15, w: 5, h: 6 },
        { i: 'recent_trades', x: 0, y: 17, w: 3, h: 6 },
        { i: 'order_management', x: 3, y: 21, w: 9, h: 6 },
        { i: 'performance', x: 0, y: 27, w: 12, h: 6 },
        { i: 'allocation', x: 0, y: 33, w: 6, h: 6 },
        { i: 'correlation', x: 6, y: 33, w: 6, h: 6 },
        { i: 'risk_settings', x: 0, y: 39, w: 12, h: 4 },
        { i: 'sentiment', x: 0, y: 43, w: 4, h: 6 },
        { i: 'prediction', x: 4, y: 43, w: 4, h: 6 },
        { i: 'news', x: 8, y: 43, w: 4, h: 6 },
        { i: 'strategy', x: 0, y: 49, w: 12, h: 8 },
        { i: 'settings_page', x: 0, y: 57, w: 12, h: 8 },
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
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end' }}>
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
