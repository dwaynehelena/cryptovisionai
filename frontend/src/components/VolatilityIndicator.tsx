import React, { useEffect, useState } from 'react';
import { Paper, Typography, Box, CircularProgress, Alert } from '@mui/material';
import { apiService } from '../services/api';

interface Props {
    symbol: string;
}

const VolatilityIndicator: React.FC<Props> = ({ symbol }) => {
    const [atr, setAtr] = useState<number | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const data = await apiService.getIndicators(symbol);
                if (data && data.length > 0) {
                    const latest = data[data.length - 1];
                    setAtr(latest.atr);
                }
                setError(null);
            } catch (err) {
                console.error('Failed to fetch indicators:', err);
                setError('Failed to load volatility data');
            } finally {
                setLoading(false);
            }
        };

        if (symbol) {
            fetchData();
        }
    }, [symbol]);

    return (
        <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Volatility ({symbol.replace('USDT', '')})</Typography>

            {loading ? (
                <Box display="flex" justifyContent="center" p={2}>
                    <CircularProgress size={24} />
                </Box>
            ) : error ? (
                <Alert severity="error">{error}</Alert>
            ) : (
                <Box>
                    <Typography variant="subtitle2" color="text.secondary">ATR (14)</Typography>
                    <Typography variant="h4" color="primary">
                        {atr ? atr.toFixed(2) : 'N/A'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        Average True Range
                    </Typography>
                </Box>
            )}
        </Paper>
    );
};

export default VolatilityIndicator;
