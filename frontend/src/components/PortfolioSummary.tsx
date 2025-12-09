import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    useTheme,
    alpha,
    Skeleton,
    Grid,
} from '@mui/material';
import {
    TrendingUp as TrendingUpIcon,
    TrendingDown as TrendingDownIcon,
    AccountBalance as BalanceIcon,
    ShowChart as ChartIcon,
} from '@mui/icons-material';
import apiService from '../services/api';

interface PortfolioData {
    initial_capital: number;
    current_capital: number;
    portfolio_value: number;
    profit_loss: number;
    profit_loss_percent: number;
    open_positions: number;
    closed_positions: number;
}

export default function PortfolioSummary() {
    const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
    const [loading, setLoading] = useState(true);
    const theme = useTheme();

    useEffect(() => {
        fetchPortfolio();
        const interval = setInterval(fetchPortfolio, 30000); // Update every 30s
        return () => clearInterval(interval);
    }, []);

    const fetchPortfolio = async () => {
        try {
            const data = await apiService.getPortfolio();
            setPortfolio(data);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching portfolio:', error);
            setLoading(false);
            // Set fallback data on error
            setPortfolio({
                initial_capital: 0,
                current_capital: 0,
                portfolio_value: 0,
                profit_loss: 0,
                profit_loss_percent: 0,
                open_positions: 0,
                closed_positions: 0,
            });
        }
    };

    const StatCard = ({ title, value, change, icon, color }: any) => (
        <Card
            sx={{
                height: '100%',
                background: `linear-gradient(135deg, ${alpha(color || theme.palette.primary.main, 0.05)} 0%, ${alpha(color || theme.palette.primary.main, 0.15)} 100%)`,
                border: `1px solid ${alpha(color || theme.palette.primary.main, 0.2)}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: `0 8px 24px ${alpha(color || theme.palette.primary.main, 0.3)}`,
                },
            }}
        >
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
                        {title}
                    </Typography>
                    <Box
                        sx={{
                            p: 1,
                            borderRadius: 2,
                            backgroundColor: alpha(color || theme.palette.primary.main, 0.2),
                        }}
                    >
                        {icon}
                    </Box>
                </Box>
                {loading ? (
                    <Skeleton variant="text" width="60%" height={40} />
                ) : (
                    <>
                        <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                            {value}
                        </Typography>
                        {change !== undefined && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                {change >= 0 ? (
                                    <TrendingUpIcon sx={{ fontSize: 18, color: theme.palette.success.main }} />
                                ) : (
                                    <TrendingDownIcon sx={{ fontSize: 18, color: theme.palette.error.main }} />
                                )}
                                <Typography
                                    variant="body2"
                                    sx={{
                                        color: change >= 0 ? theme.palette.success.main : theme.palette.error.main,
                                        fontWeight: 600,
                                    }}
                                >
                                    {change >= 0 ? '+' : ''}{change}%
                                </Typography>
                            </Box>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );

    return (
        <Grid container spacing={3}>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <StatCard
                    title="Portfolio Value"
                    value={portfolio?.portfolio_value != null ? `$${portfolio.portfolio_value.toLocaleString()}` : '-'}
                    change={portfolio?.profit_loss_percent}
                    icon={<BalanceIcon sx={{ color: theme.palette.primary.main }} />}
                    color={theme.palette.primary.main}
                />
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <StatCard
                    title="Profit / Loss"
                    value={portfolio?.profit_loss != null ? `$${portfolio.profit_loss.toLocaleString()}` : '-'}
                    change={portfolio?.profit_loss_percent}
                    icon={<ChartIcon sx={{ color: portfolio && portfolio.profit_loss >= 0 ? theme.palette.success.main : theme.palette.error.main }} />}
                    color={portfolio && portfolio.profit_loss >= 0 ? theme.palette.success.main : theme.palette.error.main}
                />
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <StatCard
                    title="Open Positions"
                    value={portfolio?.open_positions || 0}
                    icon={<TrendingUpIcon sx={{ color: theme.palette.info.main }} />}
                    color={theme.palette.info.main}
                />
            </Grid>
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <StatCard
                    title="Closed Positions"
                    value={portfolio?.closed_positions || 0}
                    icon={<ChartIcon sx={{ color: theme.palette.secondary.main }} />}
                    color={theme.palette.secondary.main}
                />
            </Grid>
        </Grid>
    );
}
