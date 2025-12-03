import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    List,
    ListItem,
    IconButton,
    Box,
    useTheme,
    alpha,
    Chip
} from '@mui/material';
import {
    Delete as DeleteIcon,
    Star as StarIcon,
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon
} from '@mui/icons-material';
import apiService from '../services/api';

interface WatchlistItem {
    symbol: string;
    price: number;
    change24h: number;
}

interface Props {
    onSelectSymbol: (symbol: string) => void;
}

export default function Watchlist({ onSelectSymbol }: Props) {
    const [watchlist, setWatchlist] = useState<string[]>(() => {
        const saved = localStorage.getItem('watchlist');
        return saved ? JSON.parse(saved) : ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'];
    });
    const [marketData, setMarketData] = useState<WatchlistItem[]>([]);
    const theme = useTheme();

    useEffect(() => {
        localStorage.setItem('watchlist', JSON.stringify(watchlist));
        fetchMarketData();
        const interval = setInterval(fetchMarketData, 5000);
        return () => clearInterval(interval);
    }, [watchlist]);

    const fetchMarketData = async () => {
        try {
            const data = await Promise.all(watchlist.map(async (symbol) => {
                const ticker = await apiService.getTicker(symbol);
                // Mocking 24h change for now as getTicker might not return it
                // In a real app, you'd call a 24h ticker endpoint
                return {
                    symbol,
                    price: parseFloat(ticker.price),
                    change24h: (Math.random() * 10) - 5 // Mock change
                };
            }));
            setMarketData(data);
        } catch (error) {
            console.error('Error fetching watchlist data:', error);
        }
    };

    const removeFromWatchlist = (symbol: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setWatchlist(prev => prev.filter(s => s !== symbol));
    };

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <StarIcon sx={{ color: theme.palette.warning.main, mr: 1 }} />
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>
                        Watchlist
                    </Typography>
                </Box>

                <List sx={{ maxHeight: 300, overflow: 'auto' }}>
                    {marketData.map((item) => (
                        <ListItem
                            key={item.symbol}
                            disablePadding
                            sx={{
                                mb: 1,
                                borderRadius: 2,
                                border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
                                '&:hover': {
                                    backgroundColor: alpha(theme.palette.action.hover, 0.1),
                                    cursor: 'pointer'
                                }
                            }}
                            onClick={() => onSelectSymbol(item.symbol)}
                            secondaryAction={
                                <IconButton
                                    edge="end"
                                    aria-label="delete"
                                    size="small"
                                    onClick={(e) => removeFromWatchlist(item.symbol, e)}
                                >
                                    <DeleteIcon fontSize="small" />
                                </IconButton>
                            }
                        >
                            <Box sx={{ p: 1.5, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <Box>
                                    <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                                        {item.symbol}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        ${item.price.toLocaleString()}
                                    </Typography>
                                </Box>
                                <Chip
                                    icon={item.change24h >= 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                                    label={`${item.change24h > 0 ? '+' : ''}${item.change24h.toFixed(2)}%`}
                                    size="small"
                                    color={item.change24h >= 0 ? 'success' : 'error'}
                                    variant="outlined"
                                    sx={{ height: 24 }}
                                />
                            </Box>
                        </ListItem>
                    ))}
                </List>
            </CardContent>
        </Card>
    );
}
