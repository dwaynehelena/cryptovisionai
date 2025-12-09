import { useState } from 'react';
import { Box, Typography, Paper, TextField, Button, MenuItem, CircularProgress, Grid } from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import apiService from '../services/api';
import BacktestResults from './BacktestResults';

export default function StrategyBuilder() {
    const [strategy, setStrategy] = useState('SMA_CROSSOVER');
    const [symbol, setSymbol] = useState('BTCUSDT');
    const [timeframe, setTimeframe] = useState('1h');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<any>(null);

    const handleRunBacktest = async () => {
        setLoading(true);
        try {
            const data = await apiService.runBacktest({
                symbol,
                strategy,
                timeframe,
                start_date: '2023-01-01',
                end_date: '2023-12-31',
                parameters: {}
            });
            setResults(data);
        } catch (error) {
            console.error('Backtest failed:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" gutterBottom>Strategy Lab</Typography>

            <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid size={{ xs: 12, md: 4 }}>
                    <TextField
                        select
                        fullWidth
                        label="Strategy"
                        value={strategy}
                        onChange={(e) => setStrategy(e.target.value)}
                        size="small"
                    >
                        <MenuItem value="SMA_CROSSOVER">SMA Crossover</MenuItem>
                        <MenuItem value="RSI_REVERSAL">RSI Reversal</MenuItem>
                        <MenuItem value="MACD_TREND">MACD Trend</MenuItem>
                    </TextField>
                </Grid>
                <Grid size={{ xs: 6, md: 4 }}>
                    <TextField
                        fullWidth
                        label="Symbol"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value)}
                        size="small"
                    />
                </Grid>
                <Grid size={{ xs: 6, md: 4 }}>
                    <TextField
                        select
                        fullWidth
                        label="Timeframe"
                        value={timeframe}
                        onChange={(e) => setTimeframe(e.target.value)}
                        size="small"
                    >
                        <MenuItem value="15m">15m</MenuItem>
                        <MenuItem value="1h">1h</MenuItem>
                        <MenuItem value="4h">4h</MenuItem>
                        <MenuItem value="1d">1d</MenuItem>
                    </TextField>
                </Grid>
            </Grid>

            <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
                onClick={handleRunBacktest}
                disabled={loading}
                fullWidth
                sx={{ mb: 2 }}
            >
                {loading ? 'Running Simulation...' : 'Run Backtest'}
            </Button>

            <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                {results && <BacktestResults results={results} />}
            </Box>
        </Paper>
    );
}
