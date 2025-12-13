import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useOutletContext } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useColorMode, ColorModeContext } from './theme';
import { Box } from '@mui/material';

// Layout
import Dashboard from './components/Dashboard';
import DashboardHome from './components/DashboardHome';

// Components
import MarketDataChart from './components/MarketDataChart';
import PortfolioSummary from './components/PortfolioSummary';
import OrderBook from './components/OrderBook';
import RecentTrades from './components/RecentTrades';
import OrderForm from './components/OrderForm';
import OrderManagement from './components/OrderManagement';
import PerformanceMetrics from './components/PerformanceMetrics';
import PortfolioAllocation from './components/PortfolioAllocation';
import RiskMonitor from './components/RiskMonitor';
import RiskSettings from './components/RiskSettings';
import CorrelationMatrix from './components/CorrelationMatrix';
import VolatilityIndicator from './components/VolatilityIndicator';
import Watchlist from './components/Watchlist';
import Settings from './components/Settings';
import SentimentAnalysis from './components/SentimentAnalysis';
import AIPrediction from './components/AIPrediction';
import NewsFeed from './components/NewsFeed';
import StrategyBuilder from './components/StrategyBuilder';
import ModelPerformanceCard from './components/ModelPerformanceCard';

// Helper to inject context into components
const RouteWrapper = ({ Component, fullHeight = false }: { Component: any, fullHeight?: boolean }) => {
  const context: any = useOutletContext();

  // Default props if context is missing (though it shouldn't be under Dashboard)
  const props = {
    symbol: context?.selectedSymbol || 'BTCUSDT',
    selectedSymbol: context?.selectedSymbol || 'BTCUSDT',
    onSelectSymbol: context?.setSelectedSymbol || (() => { }),
    currentPrice: context?.currentPrice || 0,
    accountBalance: context?.accountBalance || 0
  };

  return (
    <Box sx={{
      height: fullHeight ? 'calc(100vh - 100px)' : 'auto',
      width: '100%',
      overflow: 'auto',
      p: 1
    }}>
      <Component {...props} />
    </Box>
  );
};

function App() {
  const { theme, colorMode } = useColorMode();
  const [isDraggable, setIsDraggable] = useState(false);

  return (
    <Router>
      <ColorModeContext.Provider value={colorMode}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Routes>
            <Route path="/" element={<Dashboard />}>
              <Route index element={<DashboardHome isDraggable={isDraggable} setIsDraggable={setIsDraggable} />} />

              {/* Trading */}
              <Route path="chart" element={<RouteWrapper Component={MarketDataChart} fullHeight />} />
              <Route path="trade" element={<RouteWrapper Component={OrderForm} />} />
              <Route path="orderbook" element={<RouteWrapper Component={OrderBook} fullHeight />} />
              <Route path="recent-trades" element={<RouteWrapper Component={RecentTrades} fullHeight />} />
              <Route path="orders" element={<RouteWrapper Component={OrderManagement} />} />
              <Route path="watchlist" element={<RouteWrapper Component={Watchlist} fullHeight />} />

              {/* Risk */}
              <Route path="risk" element={<RouteWrapper Component={RiskMonitor} fullHeight />} />
              <Route path="volatility" element={<RouteWrapper Component={VolatilityIndicator} />} />
              <Route path="risk-settings" element={<RouteWrapper Component={RiskSettings} />} />

              {/* Portfolio */}
              <Route path="portfolio" element={<RouteWrapper Component={PortfolioSummary} />} />
              <Route path="performance" element={<RouteWrapper Component={PerformanceMetrics} />} />
              <Route path="allocation" element={<RouteWrapper Component={PortfolioAllocation} />} />

              {/* Analysis */}
              <Route path="correlation" element={<RouteWrapper Component={CorrelationMatrix} />} />
              <Route path="sentiment" element={<RouteWrapper Component={SentimentAnalysis} />} />
              <Route path="prediction" element={<RouteWrapper Component={AIPrediction} />} />
              <Route path="models" element={<RouteWrapper Component={ModelPerformanceCard} />} />
              <Route path="news" element={<RouteWrapper Component={NewsFeed} fullHeight />} />
              <Route path="strategy" element={<RouteWrapper Component={StrategyBuilder} fullHeight />} />

              {/* System */}
              <Route path="settings" element={<RouteWrapper Component={Settings} />} />
            </Route>
          </Routes>
        </ThemeProvider>
      </ColorModeContext.Provider>
    </Router>
  );
}

export default App;
