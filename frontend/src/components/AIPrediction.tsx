import { useState, useEffect } from 'react';
import { Box, Typography, Paper, CircularProgress, Chip, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { AutoGraph, CheckCircle, Warning } from '@mui/icons-material';
import apiService from '../services/api';

interface Props {
    symbol: string;
}

export default function AIPrediction({ symbol }: Props) {
    const [prediction, setPrediction] = useState<any>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const result = await apiService.getPrediction(symbol);
                setPrediction(result);
            } catch (error) {
                console.error('Error fetching prediction:', error);
            }
        };
        fetchData();
    }, [symbol]);

    if (!prediction) return <Paper sx={{ p: 2, display: 'flex', justifyContent: 'center' }}><CircularProgress /></Paper>;

    return (
        <Paper sx={{ p: 2, height: '100%', position: 'relative', overflow: 'hidden' }}>
            <Box sx={{ position: 'absolute', top: -10, right: -10, opacity: 0.1 }}>
                <AutoGraph sx={{ fontSize: 100 }} />
            </Box>

            <Typography variant="h6" gutterBottom>AI Forecast (24h)</Typography>

            <Box sx={{ my: 2 }}>
                <Typography variant="h3" color="primary.main" sx={{ fontWeight: 'bold' }}>
                    ${prediction.target_price.toLocaleString()}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                    <Chip
                        label={prediction.direction}
                        color={prediction.direction === 'UP' ? 'success' : prediction.direction === 'DOWN' ? 'error' : 'default'}
                        size="small"
                    />
                    <Chip
                        label={`${(prediction.confidence * 100).toFixed(0)}% Confidence`}
                        variant="outlined"
                        size="small"
                    />
                </Box>
            </Box>

            <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>Key Factors</Typography>
            <List dense>
                {prediction.factors.map((factor: string, index: number) => (
                    <ListItem key={index} disablePadding>
                        <ListItemIcon sx={{ minWidth: 30 }}>
                            {factor.includes('Risk') || factor.includes('Bearish') ?
                                <Warning fontSize="small" color="warning" /> :
                                <CheckCircle fontSize="small" color="success" />
                            }
                        </ListItemIcon>
                        <ListItemText primary={factor} />
                    </ListItem>
                ))}
            </List>
        </Paper>
    );
}
