import React, { useEffect, useState } from 'react';
import {
    Paper,
    Typography,
    Box,
    Grid,
    LinearProgress,
    Chip,
    Card,
    CardContent,
    Alert,
    IconButton
} from '@mui/material';
import {
    Warning as WarningIcon,
    CheckCircle as CheckCircleIcon,
    Error as ErrorIcon,
    Refresh as RefreshIcon
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

interface RiskStatus {
    current_drawdown: number;
    peak_value: number;
    daily_pnl: number;
    capital_used: number;
    capital_available: number;
    largest_position_pct: number;
    risk_score: number;
    trading_enabled: boolean;
    violations: string[];
    last_updated: string;
}

const RiskMonitor: React.FC = () => {
    const [status, setStatus] = useState<RiskStatus | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchRiskStatus = async () => {
        try {
            setLoading(true);
            const data = await api.getRiskStatus();
            setStatus(data);
            setError(null);
        } catch (err) {
            console.error('Failed to fetch risk status:', err);
            setError('Failed to load risk status');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchRiskStatus();
        const interval = setInterval(fetchRiskStatus, 30000); // Update every 30s
        return () => clearInterval(interval);
    }, []);

    const getRiskColor = (score: number) => {
        if (score < 30) return 'success';
        if (score < 60) return 'warning';
        return 'error';
    };

    const getDrawdownColor = (drawdown: number) => {
        if (Math.abs(drawdown) < 10) return 'success';
        if (Math.abs(drawdown) < 15) return 'warning';
        return 'error';
    };

    if (loading && !status) {
        return <LinearProgress />;
    }

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    if (!status) return null;

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" component="h2" display="flex" alignItems="center">
                    <WarningIcon sx={{ mr: 1, color: 'text.secondary' }} />
                    Risk Monitor
                </Typography>
                <Box display="flex" alignItems="center">
                    <Chip
                        label={status.trading_enabled ? "Trading Active" : "Trading Halted"}
                        color={status.trading_enabled ? "success" : "error"}
                        variant="outlined"
                        sx={{ mr: 1 }}
                    />
                    <IconButton size="small" onClick={fetchRiskStatus}>
                        <RefreshIcon />
                    </IconButton>
                </Box>
            </Box>

            <Grid container spacing={2}>
                {/* Risk Score */}
                <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                        <CardContent>
                            <Typography color="textSecondary" gutterBottom variant="caption">
                                Risk Score
                            </Typography>
                            <Box display="flex" alignItems="center" justifyContent="space-between">
                                <Typography variant="h4" color={`${getRiskColor(status.risk_score)}.main`}>
                                    {status.risk_score}/100
                                </Typography>
                                {status.risk_score < 30 ? <CheckCircleIcon color="success" /> :
                                    status.risk_score < 60 ? <WarningIcon color="warning" /> :
                                        <ErrorIcon color="error" />}
                            </Box>
                            <LinearProgress
                                variant="determinate"
                                value={status.risk_score}
                                color={getRiskColor(status.risk_score)}
                                sx={{ mt: 1, height: 8, borderRadius: 4 }}
                            />
                        </CardContent>
                    </Card>
                </Grid>

                {/* Drawdown */}
                <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                        <CardContent>
                            <Typography color="textSecondary" gutterBottom variant="caption">
                                Current Drawdown
                            </Typography>
                            <Typography variant="h4" color={getDrawdownColor(status.current_drawdown) + ".main"}>
                                {status.current_drawdown.toFixed(2)}%
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                                Peak: ${status.peak_value.toLocaleString()}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Capital Usage */}
                <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                        <CardContent>
                            <Typography color="textSecondary" gutterBottom variant="caption">
                                Capital Utilization
                            </Typography>
                            <Typography variant="h4">
                                {((status.capital_used / (status.capital_used + status.capital_available)) * 100).toFixed(1)}%
                            </Typography>
                            <Box display="flex" justifyContent="space-between" mt={1}>
                                <Typography variant="caption">Used: ${Math.round(status.capital_used).toLocaleString()}</Typography>
                                <Typography variant="caption">Free: ${Math.round(status.capital_available).toLocaleString()}</Typography>
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                {/* Largest Position */}
                <Grid item xs={12} md={3}>
                    <Card variant="outlined">
                        <CardContent>
                            <Typography color="textSecondary" gutterBottom variant="caption">
                                Largest Position
                            </Typography>
                            <Typography variant="h4" color={status.largest_position_pct > 15 ? "warning.main" : "text.primary"}>
                                {status.largest_position_pct.toFixed(1)}%
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                                Concentration Risk
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Active Violations */}
            {status.violations.length > 0 && (
                <Box mt={2}>
                    <Typography variant="subtitle2" gutterBottom color="error">
                        Active Risk Violations
                    </Typography>
                    {status.violations.map((violation, index) => (
                        <Alert severity="error" key={index} sx={{ mb: 1 }}>
                            {violation}
                        </Alert>
                    ))}
                </Box>
            )}
        </Paper>
    );
};

export default RiskMonitor;
