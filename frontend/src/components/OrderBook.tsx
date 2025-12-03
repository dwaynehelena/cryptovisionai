import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    useTheme,
    alpha,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from '@mui/material';
import apiService from '../services/api';

interface OrderBookEntry {
    price: string;
    quantity: string;
}

interface OrderBookData {
    symbol: string;
    bids: [string, string][];
    asks: [string, string][];
}

interface Props {
    symbol: string;
}

export default function OrderBook({ symbol }: Props) {
    const [orderBook, setOrderBook] = useState<OrderBookData | null>(null);
    const theme = useTheme();

    useEffect(() => {
        fetchOrderBook();
        const interval = setInterval(fetchOrderBook, 2000); // Update every 2s
        return () => clearInterval(interval);
    }, [symbol]);

    const fetchOrderBook = async () => {
        try {
            const data = await apiService.getOrderBook(symbol, 15);
            setOrderBook(data);
        } catch (error) {
            console.error('Error fetching order book:', error);
        }
    };

    if (!orderBook) {
        return null;
    }

    // Calculate total volumes
    const totalBidVolume = orderBook.bids.reduce((sum, [, qty]) => sum + parseFloat(qty), 0);
    const totalAskVolume = orderBook.asks.reduce((sum, [, qty]) => sum + parseFloat(qty), 0);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
                    Order Book - {symbol}
                </Typography>

                <Box sx={{ display: 'flex', gap: 2 }}>
                    {/* Bids (Buy Orders) */}
                    <Box sx={{ flex: 1 }}>
                        <Typography variant="subtitle2" sx={{ color: theme.palette.success.main, mb: 1, fontWeight: 600 }}>
                            Bids (Buy)
                        </Typography>
                        <TableContainer sx={{ maxHeight: 300 }}>
                            <Table size="small" stickyHeader>
                                <TableHead>
                                    <TableRow>
                                        <TableCell sx={{ fontWeight: 600 }}>Price</TableCell>
                                        <TableCell align="right" sx={{ fontWeight: 600 }}>Amount</TableCell>
                                        <TableCell align="right" sx={{ fontWeight: 600 }}>Total</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {orderBook.bids.slice(0, 15).map(([price, qty], index) => {
                                        const cumulativeVolume = orderBook.bids
                                            .slice(0, index + 1)
                                            .reduce((sum, [, q]) => sum + parseFloat(q), 0);
                                        const percentage = (cumulativeVolume / totalBidVolume) * 100;

                                        return (
                                            <TableRow
                                                key={index}
                                                sx={{
                                                    backgroundImage: `linear-gradient(to left, ${alpha(theme.palette.success.main, 0.1)} ${percentage}%, transparent ${percentage}%)`,
                                                }}
                                            >
                                                <TableCell sx={{ color: theme.palette.success.main, fontWeight: 600 }}>
                                                    ${parseFloat(price).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                                </TableCell>
                                                <TableCell align="right">{parseFloat(qty).toFixed(4)}</TableCell>
                                                <TableCell align="right">{(parseFloat(price) * parseFloat(qty)).toFixed(2)}</TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>

                    {/* Asks (Sell Orders) */}
                    <Box sx={{ flex: 1 }}>
                        <Typography variant="subtitle2" sx={{ color: theme.palette.error.main, mb: 1, fontWeight: 600 }}>
                            Asks (Sell)
                        </Typography>
                        <TableContainer sx={{ maxHeight: 300 }}>
                            <Table size="small" stickyHeader>
                                <TableHead>
                                    <TableRow>
                                        <TableCell sx={{ fontWeight: 600 }}>Price</TableCell>
                                        <TableCell align="right" sx={{ fontWeight: 600 }}>Amount</TableCell>
                                        <TableCell align="right" sx={{ fontWeight: 600 }}>Total</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {orderBook.asks.slice(0, 15).map(([price, qty], index) => {
                                        const cumulativeVolume = orderBook.asks
                                            .slice(0, index + 1)
                                            .reduce((sum, [, q]) => sum + parseFloat(q), 0);
                                        const percentage = (cumulativeVolume / totalAskVolume) * 100;

                                        return (
                                            <TableRow
                                                key={index}
                                                sx={{
                                                    backgroundImage: `linear-gradient(to left, ${alpha(theme.palette.error.main, 0.1)} ${percentage}%, transparent ${percentage}%)`,
                                                }}
                                            >
                                                <TableCell sx={{ color: theme.palette.error.main, fontWeight: 600 }}>
                                                    ${parseFloat(price).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                                                </TableCell>
                                                <TableCell align="right">{parseFloat(qty).toFixed(4)}</TableCell>
                                                <TableCell align="right">{(parseFloat(price) * parseFloat(qty)).toFixed(2)}</TableCell>
                                            </TableRow>
                                        );
                                    })}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>
                </Box>
            </CardContent>
        </Card>
    );
}
