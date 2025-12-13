import { useState, useContext } from 'react';
import { ColorModeContext } from '../theme';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
    Box,
    AppBar,
    Toolbar,
    Typography,
    IconButton,
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    useTheme,
    alpha,
    Divider,
} from '@mui/material';
import {
    Menu as MenuIcon,
    Dashboard as DashboardIcon,
    TrendingUp as TrendingUpIcon,
    AccountBalance as PortfolioIcon,
    Settings as SettingsIcon,
    ShowChart as ChartIcon,
    Brightness4 as Brightness4Icon,
    Brightness7 as Brightness7Icon,
    Security as SecurityIcon,
    ListAlt as OrderBookIcon,
    History as HistoryIcon,
    SwapHoriz as TradeIcon,
    Assessment as AssessmentIcon,
    PieChart as PieChartIcon,
    Psychology as PsychologyIcon,
    Article as NewsIcon,
    Lightbulb as StrategyIcon,
    Timeline as TimelineIcon,
    Visibility as VisibilityIcon,
    Speed as SpeedIcon,
    DataUsage as DataUsageIcon,
    Memory as MemoryIcon,
} from '@mui/icons-material';
import ConnectionStatus from './ConnectionStatus';

const drawerWidth = 240;

const Dashboard: React.FC = () => {
    const [mobileOpen, setMobileOpen] = useState(false);
    // Lifted state for symbol selection
    const [selectedSymbol, setSelectedSymbol] = useState<string>('BTCUSDT');
    const [currentPrice] = useState(0); // This should ideally come from a context or global store
    const [accountBalance] = useState(10000);

    const theme = useTheme();
    const colorMode = useContext(ColorModeContext);
    const navigate = useNavigate();
    const location = useLocation();

    const handleDrawerToggle = () => {
        setMobileOpen(!mobileOpen);
    };

    // Define icons clearly
    const SwapHorizIcon = TradeIcon;
    const ListAltIcon = OrderBookIcon;
    const ArticleIcon = NewsIcon;
    const LightbulbIcon = StrategyIcon;

    const menuItems = [
        { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
        // Trading
        { text: 'Market Chart', icon: <ChartIcon />, path: '/chart' },
        { text: 'Order Form', icon: <SwapHorizIcon />, path: '/trade' },
        { text: 'Order Book', icon: <ListAltIcon />, path: '/orderbook' },
        { text: 'Recent Trades', icon: <HistoryIcon />, path: '/recent-trades' },
        { text: 'Order Management', icon: <AssessmentIcon />, path: '/orders' },
        { text: 'Watchlist', icon: <VisibilityIcon />, path: '/watchlist' },
        // Risk
        { text: 'Risk Monitor', icon: <SecurityIcon />, path: '/risk' },
        { text: 'Volatility', icon: <SpeedIcon />, path: '/volatility' },
        { text: 'Risk Settings', icon: <SettingsIcon />, path: '/risk-settings' },
        // Portfolio
        { text: 'Portfolio Summary', icon: <PortfolioIcon />, path: '/portfolio' },
        { text: 'Performance', icon: <TimelineIcon />, path: '/performance' },
        { text: 'Allocation', icon: <PieChartIcon />, path: '/allocation' },
        // Analysis
        { text: 'Correlation', icon: <DataUsageIcon />, path: '/correlation' },
        { text: 'Sentiment', icon: <PsychologyIcon />, path: '/sentiment' },
        { text: 'AI Prediction', icon: <TrendingUpIcon />, path: '/prediction' },
        { text: 'Model Performance', icon: <MemoryIcon />, path: '/models' },
        { text: 'News Feed', icon: <ArticleIcon />, path: '/news' },
        { text: 'Strategy', icon: <LightbulbIcon />, path: '/strategy' },
        // System
        { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
    ];

    const drawer = (
        <Box>
            <Toolbar>
                <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
                    CryptoVision<span style={{ color: theme.palette.primary.main }}>AI</span>
                </Typography>
            </Toolbar>
            <Divider />
            <List>
                {menuItems.map((item) => (
                    <ListItem key={item.text} disablePadding>
                        <ListItemButton
                            selected={location.pathname === item.path}
                            onClick={() => {
                                navigate(item.path);
                                setMobileOpen(false);
                            }}
                            sx={{
                                borderRadius: 2,
                                mx: 1.5,
                                mb: 0.5,
                                '&:hover': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                },
                                '&.Mui-selected': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.2),
                                    '&:hover': {
                                        backgroundColor: alpha(theme.palette.primary.main, 0.3),
                                    },
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
                        {menuItems.find(item => item.path === location.pathname)?.text || 'Dashboard'}
                    </Typography>

                    <ConnectionStatus />
                    <IconButton sx={{ ml: 1 }} onClick={colorMode.toggleColorMode} color="inherit">
                        {theme.palette.mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
                    </IconButton>

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
                <Outlet context={{ selectedSymbol, setSelectedSymbol, currentPrice, accountBalance }} />
            </Box>
        </Box>
    );
}

export default Dashboard;
