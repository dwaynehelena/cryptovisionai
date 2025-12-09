import { useState } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    TextField,
    Button,
    ToggleButton,
    ToggleButtonGroup,
    InputAdornment,
    Alert,
    useTheme,
    alpha,
    Divider,
} from '@mui/material';
import {
    TrendingUp as BuyIcon,
    TrendingDown as SellIcon,
} from '@mui/icons-material';
import apiService from '../services/api';

interface Props {
    symbol: string;
    currentPrice: number;
}

export default function OrderForm({ symbol, currentPrice }: Props) {
    const theme = useTheme();
    const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT' | 'STOP_LOSS'>('MARKET');
    const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
    const [quantity, setQuantity] = useState('');
    const [price, setPrice] = useState('');
    const [stopPrice, setStopPrice] = useState('');
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    const handleOrderTypeChange = (_: React.MouseEvent<HTMLElement>, newType: 'MARKET' | 'LIMIT' | 'STOP_LOSS' | null) => {
        if (newType !== null) {
            setOrderType(newType);
        }
    };

    const handleSideChange = (_: React.MouseEvent<HTMLElement>, newSide: 'BUY' | 'SELL' | null) => {
        if (newSide !== null) {
            setSide(newSide);
        }
    };

    const handleSubmit = async () => {
        setLoading(true);
        setMessage(null);

        try {
            let result;
            const orderData = {
                symbol,
                side,
                order_type: orderType,
                quantity: parseFloat(quantity),
                price: price ? parseFloat(price) : undefined,
                stop_price: stopPrice ? parseFloat(stopPrice) : undefined,
            };

            if (orderType === 'MARKET') {
                result = await apiService.placeMarketOrder(orderData);
            } else if (orderType === 'LIMIT') {
                result = await apiService.placeLimitOrder(orderData);
            } else {
                result = await apiService.placeStopOrder(orderData);
            }

            setMessage({
                type: 'success',
                text: `${side} order placed successfully! Order ID: ${result.order_id}`,
            });

            // Reset form
            setQuantity('');
            setPrice('');
            setStopPrice('');
        } catch (error: any) {
            setMessage({
                type: 'error',
                text: error.response?.data?.detail || 'Failed to place order',
            });
        } finally {
            setLoading(false);
        }
    };

    const calculateTotal = () => {
        const qty = parseFloat(quantity) || 0;
        const orderPrice = orderType === 'MARKET' ? currentPrice : parseFloat(price) || 0;
        return (qty * orderPrice).toFixed(2);
    };

    return (
        <Card>
            <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
                    Place Order - {symbol}
                </Typography>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {/* Order Type Selector */}
                    <ToggleButtonGroup
                        value={orderType}
                        exclusive
                        onChange={handleOrderTypeChange}
                        fullWidth
                        size="small"
                        sx={{
                            '& .MuiToggleButton-root': {
                                borderRadius: 2,
                                '&.Mui-selected': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.2),
                                    color: theme.palette.primary.main,
                                },
                            },
                        }}
                    >
                        <ToggleButton value="MARKET">Market</ToggleButton>
                        <ToggleButton value="LIMIT">Limit</ToggleButton>
                        <ToggleButton value="STOP_LOSS">Stop Loss</ToggleButton>
                    </ToggleButtonGroup>

                    {/* Buy/Sell Selector */}
                    <ToggleButtonGroup
                        value={side}
                        exclusive
                        onChange={handleSideChange}
                        fullWidth
                        sx={{
                            '& .MuiToggleButton-root': {
                                borderRadius: 2,
                                py: 1.5,
                                '&.Mui-selected': {
                                    fontWeight: 700,
                                },
                            },
                        }}
                    >
                        <ToggleButton
                            value="BUY"
                            sx={{
                                '&.Mui-selected': {
                                    backgroundColor: alpha(theme.palette.success.main, 0.2),
                                    color: theme.palette.success.main,
                                    '&:hover': {
                                        backgroundColor: alpha(theme.palette.success.main, 0.3),
                                    },
                                },
                            }}
                        >
                            <BuyIcon sx={{ mr: 1 }} />
                            Buy
                        </ToggleButton>
                        <ToggleButton
                            value="SELL"
                            sx={{
                                '&.Mui-selected': {
                                    backgroundColor: alpha(theme.palette.error.main, 0.2),
                                    color: theme.palette.error.main,
                                    '&:hover': {
                                        backgroundColor: alpha(theme.palette.error.main, 0.3),
                                    },
                                },
                            }}
                        >
                            <SellIcon sx={{ mr: 1 }} />
                            Sell
                        </ToggleButton>
                    </ToggleButtonGroup>

                    {/* Quantity Input */}
                    <TextField
                        label="Quantity"
                        type="number"
                        value={quantity}
                        onChange={(e) => setQuantity(e.target.value)}
                        fullWidth
                        InputProps={{
                            endAdornment: <InputAdornment position="end">{symbol.replace('USDT', '')}</InputAdornment>,
                        }}
                    />

                    {/* Price Input (for Limit orders) */}
                    {orderType === 'LIMIT' && (
                        <TextField
                            label="Limit Price"
                            type="number"
                            value={price}
                            onChange={(e) => setPrice(e.target.value)}
                            fullWidth
                            InputProps={{
                                startAdornment: <InputAdornment position="start">$</InputAdornment>,
                            }}
                        />
                    )}

                    {/* Stop Price Input (for Stop Loss orders) */}
                    {orderType === 'STOP_LOSS' && (
                        <>
                            <TextField
                                label="Stop Price"
                                type="number"
                                value={stopPrice}
                                onChange={(e) => setStopPrice(e.target.value)}
                                fullWidth
                                InputProps={{
                                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                                }}
                            />
                            <TextField
                                label="Limit Price"
                                type="number"
                                value={price}
                                onChange={(e) => setPrice(e.target.value)}
                                fullWidth
                                InputProps={{
                                    startAdornment: <InputAdornment position="start">$</InputAdornment>,
                                }}
                            />
                        </>
                    )}

                    <Divider />

                    {/* Order Summary */}
                    <Box>
                        <Typography variant="body2" color="text.secondary">
                            Current Price: ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Total: ${calculateTotal()} USDT
                        </Typography>
                    </Box>

                    {/* Submit Button */}
                    <Button
                        variant="contained"
                        onClick={handleSubmit}
                        disabled={loading || !quantity || (orderType !== 'MARKET' && !price)}
                        sx={{
                            py: 1.5,
                            backgroundColor: side === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                            '&:hover': {
                                backgroundColor: side === 'BUY' ? theme.palette.success.dark : theme.palette.error.dark,
                            },
                        }}
                    >
                        {loading ? 'Placing Order...' : `${side} ${symbol}`}
                    </Button>

                    {/* Message */}
                    {message && (
                        <Alert severity={message.type} onClose={() => setMessage(null)}>
                            {message.text}
                        </Alert>
                    )}
                </Box>
            </CardContent>
        </Card>
    );
}
