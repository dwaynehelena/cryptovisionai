import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    IconButton,
    useTheme,
    alpha,
    Tabs,
    Tab,
} from '@mui/material';
import {
    Delete as DeleteIcon,
    Refresh as RefreshIcon,
} from '@mui/icons-material';
import apiService from '../services/api';

interface Order {
    order_id: number;
    symbol: string;
    side: string;
    type: string;
    quantity: string;
    executed_qty?: string;
    price?: string;
    stop_price?: string;
    status: string;
    time: number;
}

interface Props {
    symbol?: string;
}

export default function OrderManagement({ symbol }: Props) {
    const theme = useTheme();
    const [tab, setTab] = useState(0);
    const [activeOrders, setActiveOrders] = useState<Order[]>([]);
    const [orderHistory, setOrderHistory] = useState<Order[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchOrders();
        const interval = setInterval(fetchOrders, 5000); // Refresh every 5s
        return () => clearInterval(interval);
    }, [symbol, tab]);

    const fetchOrders = async () => {
        setLoading(true);
        try {
            if (tab === 0) {
                // Active orders
                const data = await apiService.getActiveOrders(symbol);
                setActiveOrders(data);
            } else {
                // Order history
                if (symbol) {
                    const data = await apiService.getOrderHistory(symbol, 50);
                    setOrderHistory(data);
                }
            }
        } catch (error) {
            console.error('Error fetching orders:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleCancelOrder = async (orderId: number, orderSymbol: string) => {
        try {
            await apiService.cancelOrder(orderId, orderSymbol);
            // Refresh orders
            fetchOrders();
        } catch (error) {
            console.error('Error canceling order:', error);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'FILLED':
                return theme.palette.success.main;
            case 'CANCELED':
                return theme.palette.error.main;
            case 'NEW':
            case 'PARTIALLY_FILLED':
                return theme.palette.warning.main;
            default:
                return theme.palette.text.secondary;
        }
    };

    const renderOrderTable = (orders: Order[], showActions: boolean = false) => (
        <TableContainer sx={{ maxHeight: 400 }}>
            <Table stickyHeader size="small">
                <TableHead>
                    <TableRow>
                        <TableCell sx={{ fontWeight: 600 }}>Symbol</TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Side</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>Quantity</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>Price</TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Time</TableCell>
                        {showActions && <TableCell align="center" sx={{ fontWeight: 600 }}>Actions</TableCell>}
                    </TableRow>
                </TableHead>
                <TableBody>
                    {orders.map((order) => (
                        <TableRow
                            key={order.order_id}
                            sx={{
                                '&:hover': {
                                    backgroundColor: alpha(theme.palette.primary.main, 0.05),
                                },
                            }}
                        >
                            <TableCell sx={{ fontWeight: 600 }}>{order.symbol}</TableCell>
                            <TableCell>{order.type}</TableCell>
                            <TableCell>
                                <Chip
                                    label={order.side}
                                    size="small"
                                    sx={{
                                        backgroundColor: alpha(
                                            order.side === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                                            0.2
                                        ),
                                        color: order.side === 'BUY' ? theme.palette.success.main : theme.palette.error.main,
                                        fontWeight: 600,
                                    }}
                                />
                            </TableCell>
                            <TableCell align="right">
                                {order.executed_qty ? `${order.executed_qty} / ${order.quantity}` : order.quantity}
                            </TableCell>
                            <TableCell align="right">
                                {order.price ? `$${parseFloat(order.price).toLocaleString()}` : 'MARKET'}
                            </TableCell>
                            <TableCell>
                                <Typography
                                    variant="body2"
                                    sx={{
                                        color: getStatusColor(order.status),
                                        fontWeight: 600,
                                    }}
                                >
                                    {order.status}
                                </Typography>
                            </TableCell>
                            <TableCell>
                                {new Date(order.time).toLocaleString([], {
                                    month: 'short',
                                    day: 'numeric',
                                    hour: '2-digit',
                                    minute: '2-digit',
                                })}
                            </TableCell>
                            {showActions && (
                                <TableCell align="center">
                                    <IconButton
                                        size="small"
                                        onClick={() => handleCancelOrder(order.order_id, order.symbol)}
                                        sx={{ color: theme.palette.error.main }}
                                    >
                                        <DeleteIcon fontSize="small" />
                                    </IconButton>
                                </TableCell>
                            )}
                        </TableRow>
                    ))}
                    {orders.length === 0 && (
                        <TableRow>
                            <TableCell colSpan={showActions ? 8 : 7} align="center" sx={{ py: 4 }}>
                                <Typography variant="body2" color="text.secondary">
                                    No orders found
                                </Typography>
                            </TableCell>
                        </TableRow>
                    )}
                </TableBody>
            </Table>
        </TableContainer>
    );

    return (
        <Card>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>
                        Order Management
                    </Typography>
                    <IconButton onClick={fetchOrders} disabled={loading}>
                        <RefreshIcon />
                    </IconButton>
                </Box>

                <Tabs
                    value={tab}
                    onChange={(e, newValue) => setTab(newValue)}
                    sx={{
                        mb: 2,
                        '& .MuiTab-root': {
                            fontWeight: 600,
                        },
                    }}
                >
                    <Tab label={`Active Orders (${activeOrders.length})`} />
                    <Tab label="Order History" disabled={!symbol} />
                </Tabs>

                {tab === 0 && renderOrderTable(activeOrders, true)}
                {tab === 1 && renderOrderTable(orderHistory, false)}
            </CardContent>
        </Card>
    );
}
