import { useState, useEffect } from 'react';
import { Box, Typography, Paper, LinearProgress, Chip, Grid } from '@mui/material';
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material';
import apiService from '../services/api';

interface Props {
    symbol: string;
}

export default function SentimentAnalysis({ symbol }: Props) {
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const result = await apiService.getSentiment(symbol);
                setData(result);
            } catch (error) {
                console.error('Error fetching sentiment:', error);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 60000);
        return () => clearInterval(interval);
    }, [symbol]);

    if (!data) return <LinearProgress />;

    const getSentimentColor = (score: number) => {
        if (score > 0.3) return 'success.main';
        if (score < -0.3) return 'error.main';
        return 'text.secondary';
    };

    const getIcon = (label: string) => {
        if (label === 'Bullish') return <TrendingUp color="success" />;
        if (label === 'Bearish') return <TrendingDown color="error" />;
        return <Remove color="disabled" />;
    };

    return (
        <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>Market Sentiment: {symbol}</Typography>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                {getIcon(data.label)}
                <Typography variant="h4" sx={{ color: getSentimentColor(data.score) }}>
                    {data.label}
                </Typography>
                <Chip label={`Score: ${data.score}`} size="small" />
            </Box>

            <Typography variant="subtitle2" gutterBottom>Fear & Greed Index</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                <Box sx={{ flexGrow: 1 }}>
                    <LinearProgress
                        variant="determinate"
                        value={data.fear_greed_index}
                        color={data.fear_greed_index > 50 ? "success" : "error"}
                        sx={{ height: 10, borderRadius: 5 }}
                    />
                </Box>
                <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{data.fear_greed_index}/100</Typography>
            </Box>

            <Typography variant="subtitle2" gutterBottom>Social Sources</Typography>
            <Grid container spacing={2}>
                {Object.entries(data.sources).map(([source, score]: [string, any]) => (
                    <Grid size={4} key={source}>
                        <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                            <Typography variant="caption" display="block" sx={{ textTransform: 'capitalize' }}>
                                {source}
                            </Typography>
                            <Typography variant="body2" sx={{ color: getSentimentColor(score), fontWeight: 'bold' }}>
                                {score > 0 ? '+' : ''}{score}
                            </Typography>
                        </Paper>
                    </Grid>
                ))}
            </Grid>
        </Paper>
    );
}
