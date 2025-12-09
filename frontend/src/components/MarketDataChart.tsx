import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    useTheme,
    alpha,
    ToggleButton,
    ToggleButtonGroup,
    Skeleton,
    Autocomplete,
    TextField,
} from '@mui/material';
import {
    Line,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    ComposedChart,
} from 'recharts';
import apiService from '../services/api';
import WebSocketClient from '../services/websocket';

interface MarketData {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    bb_upper?: number;
    bb_middle?: number;
    bb_lower?: number;
}

interface Symbol {
    symbol: string;
    baseAsset: string;
    quoteAsset: string;
}

export default function MarketDataChart() {
    const [data, setData] = useState<MarketData[]>([]);
    const [loading, setLoading] = useState(true);
    const [symbol, setSymbol] = useState('BTCUSDT');
    const [interval, setInterval] = useState('1h');
    const [symbols, setSymbols] = useState<Symbol[]>([]);
    const [showIndicators, setShowIndicators] = useState(false);
    const [, setWsClient] = useState<WebSocketClient | null>(null);
    const [livePrice, setLivePrice] = useState<number | null>(null);
    const theme = useTheme();

    // Fetch available symbols on mount
    useEffect(() => {
        fetchSymbols();
    }, []);

    // Fetch market data when symbol or interval changes
    useEffect(() => {
        fetchMarketData();
    }, [symbol, interval]);

    // Setup WebSocket for real-time price updates
    useEffect(() => {
        const ws = new WebSocketClient(symbol);
        ws.connect();

        const unsubscribe = ws.onMessage((message) => {
            if (message.type === 'price_update') {
                setLivePrice(parseFloat(message.price));
            }
        });

        setWsClient(ws);

        return () => {
            unsubscribe();
            ws.disconnect();
        };
    }, [symbol]);

    const fetchSymbols = async () => {
        try {
            const symbolsData = await apiService.getSymbols();
            // Filter for USDT pairs only
            const usdtPairs = symbolsData.filter((s: Symbol) => s.quoteAsset === 'USDT');
            setSymbols(usdtPairs);
        } catch (error) {
            console.error('Error fetching symbols:', error);
        }
    };

    const fetchMarketData = async () => {
        setLoading(true);
        try {
            if (showIndicators) {
                // Fetch data with indicators
                const indicatorData = await apiService.getIndicators(symbol, interval, 100);
                setData(indicatorData);
            } else {
                // Fetch regular market data
                const marketData = await apiService.getMarketData(symbol, interval, 100);
                setData(marketData);
            }
            setLoading(false);
        } catch (error) {
            console.error('Error fetching market data:', error);
            setLoading(false);
        }
    };

    const handleIntervalChange = (_: React.MouseEvent<HTMLElement>, newInterval: string | null) => {
        if (newInterval !== null) {
            setInterval(newInterval);
        }
    };

    const handleSymbolChange = (_: any, newValue: Symbol | null) => {
        if (newValue) {
            setSymbol(newValue.symbol);
        }
    };

    const toggleIndicators = () => {
        setShowIndicators(!showIndicators);
        fetchMarketData();
    };

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            return (
                <Box
                    sx={{
                        backgroundColor: alpha(theme.palette.background.paper, 0.95),
                        border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                        borderRadius: 2,
                        p: 2,
                    }}
                >
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                        {new Date(payload[0].payload.timestamp).toLocaleString()}
                    </Typography>
                    <Typography variant="body2" sx={{ color: theme.palette.primary.main }}>
                        Price: ${parseFloat(payload[0].value).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </Typography>
                    <Typography variant="body2" sx={{ color: theme.palette.secondary.main }}>
                        Volume: {(payload[0].payload.volume / 1000000).toFixed(2)}M
                    </Typography>
                    {showIndicators && payload[0].payload.rsi && (
                        <>
                            <Typography variant="body2" sx={{ color: theme.palette.warning.main, mt: 1 }}>
                                RSI: {payload[0].payload.rsi.toFixed(2)}
                            </Typography>
                            <Typography variant="body2" sx={{ color: theme.palette.info.main }}>
                                MACD: {payload[0].payload.macd?.toFixed(2)}
                            </Typography>
                        </>
                    )}
                </Box>
            );
        }
        return null;
    };

    return (
        <Card>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box>
                            <Typography variant="h6" sx={{ fontWeight: 700 }}>
                                Market Data - {symbol}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                {livePrice && (
                                    <span style={{ color: theme.palette.success.main, fontWeight: 600 }}>
                                        Live: ${livePrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                    </span>
                                )}
                            </Typography>
                        </Box>

                        <Autocomplete
                            value={symbols.find(s => s.symbol === symbol) || null}
                            onChange={handleSymbolChange}
                            options={symbols}
                            getOptionLabel={(option) => `${option.baseAsset}/${option.quoteAsset}`}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                    label="Symbol"
                                    variant="outlined"
                                    size="small"
                                    sx={{ width: 200 }}
                                />
                            )}
                            sx={{
                                '& .MuiOutlinedInput-root': {
                                    borderRadius: 2,
                                },
                            }}
                        />
                    </Box>

                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <ToggleButtonGroup
                            value={interval}
                            exclusive
                            onChange={handleIntervalChange}
                            size="small"
                            sx={{
                                '& .MuiToggleButton-root': {
                                    borderRadius: 2,
                                    border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                                    color: theme.palette.text.secondary,
                                    '&.Mui-selected': {
                                        backgroundColor: alpha(theme.palette.primary.main, 0.2),
                                        color: theme.palette.primary.main,
                                        '&:hover': {
                                            backgroundColor: alpha(theme.palette.primary.main, 0.3),
                                        },
                                    },
                                },
                            }}
                        >
                            <ToggleButton value="15m">15m</ToggleButton>
                            <ToggleButton value="1h">1h</ToggleButton>
                            <ToggleButton value="4h">4h</ToggleButton>
                            <ToggleButton value="1d">1d</ToggleButton>
                        </ToggleButtonGroup>

                        <ToggleButton
                            value="indicators"
                            selected={showIndicators}
                            onChange={toggleIndicators}
                            size="small"
                            sx={{
                                borderRadius: 2,
                                border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                                '&.Mui-selected': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.2),
                                    color: theme.palette.primary.main,
                                },
                            }}
                        >
                            Indicators
                        </ToggleButton>
                    </Box>
                </Box>

                {loading ? (
                    <Skeleton variant="rectangular" height={400} sx={{ borderRadius: 2 }} />
                ) : (
                    <ResponsiveContainer width="100%" height={400}>
                        {showIndicators ? (
                            <ComposedChart data={data}>
                                <defs>
                                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.text.primary, 0.1)} />
                                <XAxis
                                    dataKey="timestamp"
                                    tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    stroke={theme.palette.text.secondary}
                                    style={{ fontSize: '12px' }}
                                />
                                <YAxis
                                    yAxisId="price"
                                    domain={['auto', 'auto']}
                                    stroke={theme.palette.text.secondary}
                                    style={{ fontSize: '12px' }}
                                    tickFormatter={(value) => `$${value.toLocaleString()}`}
                                />
                                <Tooltip content={<CustomTooltip />} />
                                <Legend />
                                <Area
                                    yAxisId="price"
                                    type="monotone"
                                    dataKey="close"
                                    stroke={theme.palette.primary.main}
                                    strokeWidth={2}
                                    fill="url(#colorPrice)"
                                    name="Price"
                                />
                                {data[0]?.bb_upper && (
                                    <>
                                        <Line yAxisId="price" type="monotone" dataKey="bb_upper" stroke={theme.palette.warning.main} strokeWidth={1} dot={false} name="BB Upper" />
                                        <Line yAxisId="price" type="monotone" dataKey="bb_middle" stroke={theme.palette.info.main} strokeWidth={1} dot={false} name="BB Middle" />
                                        <Line yAxisId="price" type="monotone" dataKey="bb_lower" stroke={theme.palette.warning.main} strokeWidth={1} dot={false} name="BB Lower" />
                                    </>
                                )}
                            </ComposedChart>
                        ) : (
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.3} />
                                        <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke={alpha(theme.palette.text.primary, 0.1)} />
                                <XAxis
                                    dataKey="timestamp"
                                    tickFormatter={(value) => new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    stroke={theme.palette.text.secondary}
                                    style={{ fontSize: '12px' }}
                                />
                                <YAxis
                                    domain={['auto', 'auto']}
                                    stroke={theme.palette.text.secondary}
                                    style={{ fontSize: '12px' }}
                                    tickFormatter={(value) => `$${value.toLocaleString()}`}
                                />
                                <Tooltip content={<CustomTooltip />} />
                                <Area
                                    type="monotone"
                                    dataKey="close"
                                    stroke={theme.palette.primary.main}
                                    strokeWidth={2}
                                    fill="url(#colorPrice)"
                                />
                            </AreaChart>
                        )}
                    </ResponsiveContainer>
                )}
            </CardContent>
        </Card>
    );
}
