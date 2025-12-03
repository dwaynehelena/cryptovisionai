import React, { useEffect, useState } from 'react';
import {
    Paper,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tooltip,
    CircularProgress,
    Alert
} from '@mui/material';
import { apiService } from '../services/api';

interface CorrelationData {
    correlation_matrix: Record<string, Record<string, number>>;
    high_correlations: Array<{
        symbol1: string;
        symbol2: string;
        correlation: number;
    }>;
    message?: string;
}

const CorrelationMatrix: React.FC = () => {
    const [data, setData] = useState<CorrelationData | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const result = await apiService.getCorrelationMatrix();
                setData(result);
                setError(null);
            } catch (err) {
                console.error('Failed to fetch correlation matrix:', err);
                setError('Failed to load correlation data');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        // Refresh every minute
        const interval = setInterval(fetchData, 60000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Paper sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
                <CircularProgress />
            </Paper>
        );
    }

    if (error) {
        return (
            <Paper sx={{ p: 2 }}>
                <Alert severity="error">{error}</Alert>
            </Paper>
        );
    }

    if (!data || !data.correlation_matrix || Object.keys(data.correlation_matrix).length === 0) {
        return (
            <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>Asset Correlations</Typography>
                <Alert severity="info">
                    {data?.message || "Not enough positions to calculate correlation (need at least 2)."}
                </Alert>
            </Paper>
        );
    }

    const symbols = Object.keys(data.correlation_matrix);

    const getCellColor = (value: number) => {
        // Red for high positive correlation, Blue for high negative, White/Grey for neutral
        if (value === 1) return 'rgba(0, 0, 0, 0.05)'; // Self correlation

        const opacity = Math.abs(value);
        if (value > 0) {
            return `rgba(244, 67, 54, ${opacity * 0.8})`; // Red
        } else {
            return `rgba(33, 150, 243, ${opacity * 0.8})`; // Blue
        }
    };

    return (
        <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Asset Correlations</Typography>

            {data.high_correlations.length > 0 && (
                <Alert severity="warning" sx={{ mb: 2 }}>
                    High correlation detected between: {data.high_correlations.map(c =>
                        `${c.symbol1} & ${c.symbol2} (${c.correlation})`
                    ).join(', ')}
                </Alert>
            )}

            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell></TableCell>
                            {symbols.map(symbol => (
                                <TableCell key={symbol} align="center" sx={{ fontWeight: 'bold' }}>
                                    {symbol.replace('USDT', '')}
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {symbols.map(rowSymbol => (
                            <TableRow key={rowSymbol}>
                                <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                                    {rowSymbol.replace('USDT', '')}
                                </TableCell>
                                {symbols.map(colSymbol => {
                                    const value = data.correlation_matrix[rowSymbol][colSymbol];
                                    return (
                                        <Tooltip key={colSymbol} title={`${rowSymbol} vs ${colSymbol}: ${value.toFixed(3)}`}>
                                            <TableCell
                                                align="center"
                                                sx={{
                                                    bgcolor: getCellColor(value),
                                                    color: Math.abs(value) > 0.5 ? '#fff' : 'inherit',
                                                    fontWeight: Math.abs(value) > 0.7 ? 'bold' : 'normal'
                                                }}
                                            >
                                                {value.toFixed(2)}
                                            </TableCell>
                                        </Tooltip>
                                    );
                                })}
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
};

export default CorrelationMatrix;
