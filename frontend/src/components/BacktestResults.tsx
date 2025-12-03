import { Box, Typography, Grid, Paper } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface Props {
    results: {
        total_return: number;
        max_drawdown: number;
        win_rate: number;
        trades_count: number;
        equity_curve: any[];
    };
}

export default function BacktestResults({ results }: Props) {
    return (
        <Box>
            <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid size={6}>
                    <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="caption">Total Return</Typography>
                        <Typography variant="h6" color={results.total_return >= 0 ? 'success.main' : 'error.main'}>
                            {results.total_return}%
                        </Typography>
                    </Paper>
                </Grid>
                <Grid size={6}>
                    <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="caption">Win Rate</Typography>
                        <Typography variant="h6">{results.win_rate}%</Typography>
                    </Paper>
                </Grid>
                <Grid size={6}>
                    <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="caption">Max Drawdown</Typography>
                        <Typography variant="h6" color="error.main">-{results.max_drawdown}%</Typography>
                    </Paper>
                </Grid>
                <Grid size={6}>
                    <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
                        <Typography variant="caption">Trades</Typography>
                        <Typography variant="h6">{results.trades_count}</Typography>
                    </Paper>
                </Grid>
            </Grid>

            <Box sx={{ height: 200, width: '100%' }}>
                <ResponsiveContainer>
                    <LineChart data={results.equity_curve}>
                        <XAxis dataKey="time" hide />
                        <YAxis domain={['auto', 'auto']} hide />
                        <Tooltip
                            labelFormatter={(label) => new Date(label).toLocaleDateString()}
                            formatter={(value: number) => [`$${value.toFixed(2)}`, 'Equity']}
                        />
                        <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#8884d8"
                            strokeWidth={2}
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </Box>
        </Box>
    );
}
