import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    useTheme,
    alpha,
    List,
    ListItem,
    ListItemText,
    Chip,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import apiService from '../services/api';

interface Trade {
    id: number;
    price: string;
    qty: string;
    time: number;
    isBuyerMaker: boolean;
}

interface Props {
    symbol: string;
}

export default function RecentTrades({ symbol }: Props) {
    const [trades, setTrades] = useState<Trade[]>([]);
    const theme = useTheme();

    useEffect(() => {
        fetchTrades();
        const interval = setInterval(fetchTrades, 2000); // Update every 2s
        return () => clearInterval(interval);
    }, [symbol]);

    const fetchTrades = async () => {
        try {
            const data = await apiService.getRecentTrades(symbol, 30);
            setTrades(data);
        } catch (error) {
            console.error('Error fetching recent trades:', error);
        }
    };

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
                    Recent Trades - {symbol}
                </Typography>

                <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                    {trades.map((trade) => {
                        const isBuy = !trade.isBuyerMaker;
                        const time = new Date(trade.time).toLocaleTimeString();

                        return (
                            <ListItem
                                key={trade.id}
                                sx={{
                                    borderLeft: `3px solid ${isBuy ? theme.palette.success.main : theme.palette.error.main}`,
                                    mb: 0.5,
                                    borderRadius: 1,
                                    backgroundColor: alpha(
                                        isBuy ? theme.palette.success.main : theme.palette.error.main,
                                        0.05
                                    ),
                                    transition: 'all 0.2s',
                                    '&:hover': {
                                        backgroundColor: alpha(
                                            isBuy ? theme.palette.success.main : theme.palette.error.main,
                                            0.1
                                        ),
                                    },
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 2 }}>
                                    {isBuy ? (
                                        <TrendingUpIcon sx={{ color: theme.palette.success.main }} />
                                    ) : (
                                        <TrendingDownIcon sx={{ color: theme.palette.error.main }} />
                                    )}

                                    <ListItemText
                                        primary={
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <Typography
                                                    variant="body2"
                                                    sx={{
                                                        fontWeight: 700,
                                                        color: isBuy ? theme.palette.success.main : theme.palette.error.main,
                                                    }}
                                                >
                                                    ${parseFloat(trade.price).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                                </Typography>
                                                <Chip
                                                    label={isBuy ? 'BUY' : 'SELL'}
                                                    size="small"
                                                    sx={{
                                                        backgroundColor: alpha(
                                                            isBuy ? theme.palette.success.main : theme.palette.error.main,
                                                            0.2
                                                        ),
                                                        color: isBuy ? theme.palette.success.main : theme.palette.error.main,
                                                        fontWeight: 600,
                                                        fontSize: '0.7rem',
                                                    }}
                                                />
                                            </Box>
                                        }
                                        secondary={
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                                                <Typography variant="caption" color="text.secondary">
                                                    Amount: {parseFloat(trade.qty).toFixed(4)}
                                                </Typography>
                                                <Typography variant="caption" color="text.secondary">
                                                    {time}
                                                </Typography>
                                            </Box>
                                        }
                                        secondaryTypographyProps={{ component: 'div' }}
                                    />
                                </Box>
                            </ListItem>
                        );
                    })}
                </List>
            </CardContent>
        </Card>
    );
}
