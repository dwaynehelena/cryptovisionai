import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    Grid,
    useTheme,
    alpha,
    LinearProgress,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    ShowChart as ChartIcon,
    AccountBalance as BalanceIcon,
} from '@mui/icons-material';
import apiService from '../services/api';

interface PerformanceMetrics {
    sharpe_ratio: number;
    max_drawdown: number;
    max_drawdown_pct: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
    profit_factor: number | string;
    total_pnl: number;
}

export default function PerformanceMetrics() {
    const theme = useTheme();
    const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchMetrics();
    }, []);

    const fetchMetrics = async () => {
        setLoading(true);
        try {
            const data = await apiService.getPerformanceMetrics();
            setMetrics(data);
        } catch (error) {
            console.error('Error fetching performance metrics:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading || !metrics) {
        return (
            <Card>
                <CardContent>
                    <Typography variant="h6" sx={{ mb: 2 }}>Performance Metrics</Typography>
                    <LinearProgress />
                </CardContent>
            </Card>
        );
    }

    const MetricCard = ({ title, value, subtitle, icon, color }: any) => (
        <Box
            sx={{
                p: 2,
                borderRadius: 2,
                backgroundColor: alpha(color || theme.palette.primary.main, 0.1),
                border: `1px solid ${alpha(color || theme.palette.primary.main, 0.3)}`,
            }}
        >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                {icon}
                <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                    {title}
                </Typography>
            </Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color }}>
                {value}
            </Typography>
            {subtitle && (
                <Typography variant="caption" color="text.secondary">
                    {subtitle}
                </Typography>
            )}
        </Box>
    );

    const winRateColor = metrics.win_rate >= 50 ? theme.palette.success.main : theme.palette.error.main;
    const sharpeColor = metrics.sharpe_ratio >= 1 ? theme.palette.success.main : theme.palette.warning.main;
    const pnlColor = metrics.total_pnl >= 0 ? theme.palette.success.main : theme.palette.error.main;

    return (
        <Card>
            <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>
                    Performance Analytics
                </Typography>

                <Grid container spacing={2}>
                    {/* Sharpe Ratio */}
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Sharpe Ratio"
                            value={metrics.sharpe_ratio.toFixed(2)}
                            subtitle="Risk-adjusted returns"
                            icon={<ChartIcon sx={{ color: sharpeColor }} />}
                            color={sharpeColor}
                        />
                    </Grid>

                    {/* Max Drawdown */}
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Max Drawdown"
                            value={`${metrics.max_drawdown_pct.toFixed(2)}%`}
                            subtitle={`$${metrics.max_drawdown.toFixed(2)}`}
                            icon={<TrendingDownIcon sx={{ color: theme.palette.error.main }} />}
                            color={theme.palette.error.main}
                        />
                    </Grid>

                    {/* Win Rate */}
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Win Rate"
                            value={`${metrics.win_rate.toFixed(1)}%`}
                            subtitle={`${metrics.winning_trades}W / ${metrics.losing_trades}L`}
                            icon={<TrendingUpIcon sx={{ color: winRateColor }} />}
                            color={winRateColor}
                        />
                    </Grid>

                    {/* Profit Factor */}
                    <Grid item xs={12} sm={6} md={3}>
                        <MetricCard
                            title="Profit Factor"
                            value={typeof metrics.profit_factor === 'number' ? metrics.profit_factor.toFixed(2) : '∞'}
                            subtitle="Total wins / Total losses"
                            icon={<BalanceIcon sx={{ color: theme.palette.primary.main }} />}
                            color={theme.palette.primary.main}
                        />
                    </Grid>

                    {/* Total Trades */}
                    <Grid item xs={12} sm={6} md={3}>
                        <Box
                            sx={{
                                p: 2,
                                borderRadius: 2,
                                backgroundColor: alpha(theme.palette.info.main, 0.1),
                                border: `1px solid ${alpha(theme.palette.info.main, 0.3)}`,
                            }}
                        >
                            <Typography variant="caption" color="text.secondary">
                                Total Trades
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700 }}>
                                {metrics.total_trades}
                            </Typography>
                        </Box>
                    </Grid>

                    {/* Average Win */}
                    <Grid item xs={12} sm={6} md={3}>
                        <Box
                            sx={{
                                p: 2,
                                borderRadius: 2,
                                backgroundColor: alpha(theme.palette.success.main, 0.1),
                                border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
                            }}
                        >
                            <Typography variant="caption" color="text.secondary">
                                Avg Win
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: theme.palette.success.main }}>
                                ${metrics.avg_win.toFixed(2)}
                            </Typography>
                        </Box>
                    </Grid>

                    {/* Average Loss */}
                    <Grid item xs={12} sm={6} md={3}>
                        <Box
                            sx={{
                                p: 2,
                                borderRadius: 2,
                                backgroundColor: alpha(theme.palette.error.main, 0.1),
                                border: `1px solid ${alpha(theme.palette.error.main, 0.3)}`,
                            }}
                        >
                            <Typography variant="caption" color="text.secondary">
                                Avg Loss
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: theme.palette.error.main }}>
                                ${metrics.avg_loss.toFixed(2)}
                            </Typography>
                        </Box>
                    </Grid>

                    {/* Total P&L */}
                    <Grid item xs={12} sm={6} md={3}>
                        <Box
                            sx={{
                                p: 2,
                                borderRadius: 2,
                                backgroundColor: alpha(pnlColor, 0.1),
                                border: `1px solid ${alpha(pnlColor, 0.3)}`,
                            }}
                        >
                            <Typography variant="caption" color="text.secondary">
                                Total P&L
                            </Typography>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: pnlColor }}>
                                ${metrics.total_pnl.toFixed(2)}
                            </Typography>
                        </Box>
                    </Grid>
                </Grid>
            </CardContent>
        </Card>
    );
}
