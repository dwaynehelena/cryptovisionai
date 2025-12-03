import { useState } from 'react';
import {
    Box,
    AppBar,
    Toolbar,
    Typography,
    Container,
    Grid,
    IconButton,
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    useTheme,
    alpha,
} from '@mui/material';
import {
    Menu as MenuIcon,
    Dashboard as DashboardIcon,
    TrendingUp as TrendingUpIcon,
    AccountBalance as PortfolioIcon,
    Settings as SettingsIcon,
    ShowChart as ChartIcon,
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

const drawerWidth = 240;

const Dashboard: React.FC = () => {
    const [mobileOpen, setMobileOpen] = useState(false);
    const [selectedSymbol, setSelectedSymbol] = useState<string>('BTCUSDT');
    const [currentPrice, setCurrentPrice] = useState(0);
    const [accountBalance, setAccountBalance] = useState(10000);
    const [riskCalcOpen, setRiskCalcOpen] = useState(false);
    const theme = useTheme();

    const handleDrawerToggle = () => {
        setMobileOpen(!mobileOpen);
    };

    const menuItems = [
        { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
        { text: 'Market', icon: <TrendingUpIcon />, path: '/market' },
        { text: 'Portfolio', icon: <PortfolioIcon />, path: '/portfolio' },
        { text: 'Charts', icon: <ChartIcon />, path: '/charts' },
        { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
    ];

    const drawer = (
        <Box>
            <Toolbar>
                <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
                    CryptoVision<span style={{ color: theme.palette.primary.main }}>AI</span>
                </Typography>
            </Toolbar>
            <List>
                {menuItems.map((item) => (
                    <ListItem key={item.text} disablePadding>
                        <ListItemButton
                            sx={{
                                borderRadius: 2,
                                mx: 1.5,
                                mb: 0.5,
                                '&:hover': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                },
                            }}
                        >
                            <ListItemIcon sx={{ color: theme.palette.primary.main }}>
                                {item.icon}
                            </ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );

    return (
        <Box sx={{ display: 'flex' }}>
            <AppBar
                position="fixed"
                sx={{
                    width: { sm: `calc(100% - ${drawerWidth}px)` },
                    ml: { sm: `${drawerWidth}px` },
                }}
            >
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { sm: 'none' } }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                        Dashboard
                    </Typography>
                    <Box
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            px: 2,
                            py: 0.5,
                            borderRadius: 3,
                            backgroundColor: alpha(theme.palette.success.main, 0.15),
                        }}
                    >
                        <Box
                            sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                backgroundColor: theme.palette.success.main,
                                animation: 'pulse 2s ease-in-out infinite',
                                '@keyframes pulse': {
                                    '0%, 100%': { opacity: 1 },
                                    '50%': { opacity: 0.5 },
                                },
                            }}
                        />
                        <Typography variant="body2" sx={{ color: theme.palette.success.main, fontWeight: 600 }}>
                            Connected
                        </Typography>
                    </Box>
                </Toolbar>
            </AppBar>
            <Box
                component="nav"
                sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
            >
                <Drawer
                    variant="temporary"
                    open={mobileOpen}
                    onClose={handleDrawerToggle}
                    ModalProps={{ keepMounted: true }}
                    sx={{
                        display: { xs: 'block', sm: 'none' },
                        '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                    }}
                >
                    {drawer}
                </Drawer>
                <Drawer
                    variant="permanent"
                    sx={{
                        display: { xs: 'none', sm: 'block' },
                        '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                    }}
                    open
                >
                    {drawer}
                </Drawer>
            </Box>
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: 3,
                    width: { sm: `calc(100% - ${drawerWidth}px)` },
                    backgroundColor: theme.palette.background.default,
                    minHeight: '100vh',
                }}
            >
                <Toolbar />
                <RiskAlerts />
                <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                    <Grid container spacing={3}>
                        {/* Risk Monitor (Full Width) */}
                        <Grid size={12}>
                            <RiskMonitor />
                        </Grid>

                        {/* Portfolio Summary */}
                        <Grid size={12}>
                            <PortfolioSummary />
                        </Grid>
                        <Grid size={{ xs: 12, md: 3 }}>
                            <VolatilityIndicator symbol={selectedSymbol} />
                        </Grid>
                        <Grid size={{ xs: 12, md: 9 }}>
                            <MarketDataChart />
                        </Grid>
                        <Grid size={{ xs: 12, md: 4 }}>
                            <OrderForm symbol={selectedSymbol} currentPrice={currentPrice || 50000} />
                        </Grid>
                        <Grid size={{ xs: 12, md: 4 }}>
                            <OrderBook symbol={selectedSymbol} />
                        </Grid>
                        <Grid size={{ xs: 12, md: 4 }}>
                            <RecentTrades symbol={selectedSymbol} />
                        </Grid>
                        <Grid size={12}>
                            <OrderManagement symbol={selectedSymbol} />
                        </Grid>
                        <Grid size={12}>
                            <PerformanceMetrics />
                        </Grid>
                        <Grid size={{ xs: 12, md: 6 }}>
                            <PortfolioAllocation />
                        </Grid>
                        <Grid size={{ xs: 12, md: 6 }}>
                            <CorrelationMatrix />
                        </Grid>

                        {/* Risk Settings (Collapsible or Bottom) */}
                        <Grid size={12}>
                            <RiskSettings />
                        </Grid>
                    </Grid>
                </Container>

                <RiskCalculator
                    open={riskCalcOpen}
                    onClose={() => setRiskCalcOpen(false)}
                    currentPrice={currentPrice || 50000}
                    accountBalance={accountBalance}
                    symbol={selectedSymbol}
                />
            </Box>
        </Box>
    );
}

export default Dashboard;
