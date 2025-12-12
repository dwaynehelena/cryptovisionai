import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useColorMode, ColorModeContext } from './theme';
import Dashboard from './components/Dashboard';
import DashboardHome from './components/DashboardHome';
import Settings from './components/Settings';
import MarketDataChart from './components/MarketDataChart';
import PortfolioSummary from './components/PortfolioSummary';

// Simple placeholder components for routes that don't have dedicated pages yet
const MarketPage = () => (
  <div style={{ padding: 20 }}>
    <h2>Market Overview</h2>
    <MarketDataChart />
  </div>
);

const PortfolioPage = () => (
  <div style={{ padding: 20 }}>
    <h2>Portfolio</h2>
    <PortfolioSummary />
  </div>
);

const ChartsPage = () => (
  <div style={{ padding: 20 }}>
    <h2>Advanced Charts</h2>
    <MarketDataChart />
  </div>
);

function App() {
  const { theme, colorMode } = useColorMode();
  const [isDraggable, setIsDraggable] = useState(false);

  // We need useState imported
  return (
    <Router>
      <ColorModeContext.Provider value={colorMode}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Routes>
            <Route path="/" element={<Dashboard />}>
              <Route index element={<DashboardHome isDraggable={isDraggable} setIsDraggable={setIsDraggable} />} />
              <Route path="market" element={<MarketPage />} />
              <Route path="portfolio" element={<PortfolioPage />} />
              <Route path="charts" element={<ChartsPage />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
        </ThemeProvider>
      </ColorModeContext.Provider>
    </Router>
  );
}

export default App;
