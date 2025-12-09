import { useState, useEffect } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    TextField,
    Typography,
    Box,
    InputAdornment,
    Divider,
    useTheme,
    alpha,
    Slider,
} from '@mui/material';
import { Calculate as CalculateIcon } from '@mui/icons-material';
import apiService from '../services/api';

interface Props {
    open: boolean;
    onClose: () => void;
    currentPrice: number;
    accountBalance: number;
    symbol: string;
}

interface RiskCalcResult {
    position_size: number;
    position_value: number;
    risk_amount: number;
    risk_percent: number;
    risk_per_share: number;
    reward_ratios: {
        '1:1': number;
        '1:2': number;
        '1:3': number;
    };
}

export default function RiskCalculator({ open, onClose, currentPrice, accountBalance, symbol }: Props) {
    const theme = useTheme();
    const [entryPrice, setEntryPrice] = useState(currentPrice.toString());
    const [stopLoss, setStopLoss] = useState('');
    const [riskPercent, setRiskPercent] = useState(2);
    const [result, setResult] = useState<RiskCalcResult | null>(null);
    const [atr, setAtr] = useState<number | null>(null);

    useEffect(() => {
        if (open && symbol) {
            fetchAtr();
            setEntryPrice(currentPrice.toString());
        }
    }, [open, symbol, currentPrice]);

    const fetchAtr = async () => {
        try {
            const data = await apiService.getIndicators(symbol);
            if (data && data.length > 0) {
                const latest = data[data.length - 1];
                setAtr(latest.atr);
            }
        } catch (error) {
            console.error('Failed to fetch ATR:', error);
        }
    };

    const handleAutoStopLoss = () => {
        if (atr && entryPrice) {
            const price = parseFloat(entryPrice);
            // Default multiplier 2.0, assuming Long position for now
            // Ideally we should know the side, but let's assume Long if stop < entry or just subtract
            const stop = price - (atr * 2);
            setStopLoss(stop.toFixed(2));
        }
    };

    const handleCalculate = async () => {
        try {
            const data = await apiService.calculateRisk(
                parseFloat(entryPrice),
                parseFloat(stopLoss),
                accountBalance,
                riskPercent
            );
            setResult(data);
        } catch (error) {
            console.error('Error calculating risk:', error);
        }
    };

    const handleClose = () => {
        setResult(null);
        onClose();
    };

    return (
        <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
            <DialogTitle>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <CalculateIcon />
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>
                        Risk Calculator - {symbol}
                    </Typography>
                </Box>
            </DialogTitle>

            <DialogContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
                    {/* Account Balance */}
                    <TextField
                        label="Account Balance"
                        value={accountBalance.toFixed(2)}
                        disabled
                        fullWidth
                        InputProps={{
                            startAdornment: <InputAdornment position="start">$</InputAdornment>,
                        }}
                    />

                    {/* Entry Price */}
                    <TextField
                        label="Entry Price"
                        type="number"
                        value={entryPrice}
                        onChange={(e) => setEntryPrice(e.target.value)}
                        fullWidth
                        InputProps={{
                            startAdornment: <InputAdornment position="start">$</InputAdornment>,
                        }}
                    />

                    {/* Stop Loss */}
                    <Box sx={{ display: 'flex', gap: 1 }}>
                        <TextField
                            label="Stop Loss Price"
                            type="number"
                            value={stopLoss}
                            onChange={(e) => setStopLoss(e.target.value)}
                            fullWidth
                            InputProps={{
                                startAdornment: <InputAdornment position="start">$</InputAdornment>,
                            }}
                        />
                        <Button
                            variant="outlined"
                            onClick={handleAutoStopLoss}
                            disabled={!atr}
                            sx={{ minWidth: 100 }}
                        >
                            Auto (2xATR)
                        </Button>
                    </Box>

                    {/* Risk Percentage Slider */}
                    <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                            Risk Per Trade: {riskPercent}%
                        </Typography>
                        <Slider
                            value={riskPercent}
                            onChange={(_, value) => setRiskPercent(value as number)}
                            min={0.5}
                            max={5}
                            step={0.5}
                            marks
                            valueLabelDisplay="auto"
                            sx={{
                                '& .MuiSlider-thumb': {
                                    backgroundColor: theme.palette.primary.main,
                                },
                                '& .MuiSlider-track': {
                                    backgroundColor: theme.palette.primary.main,
                                },
                            }}
                        />
                    </Box>

                    <Button
                        variant="contained"
                        onClick={handleCalculate}
                        disabled={!entryPrice || !stopLoss}
                        fullWidth
                        sx={{ py: 1.5 }}
                    >
                        Calculate Position Size
                    </Button>

                    {/* Results */}
                    {result && (
                        <>
                            <Divider sx={{ my: 1 }} />

                            <Box
                                sx={{
                                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                                    borderRadius: 2,
                                    p: 2,
                                }}
                            >
                                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, color: theme.palette.primary.main }}>
                                    Position Sizing Results
                                </Typography>

                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Position Size:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            {result.position_size.toFixed(4)} units
                                        </Typography>
                                    </Box>

                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Position Value:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            ${result.position_value.toLocaleString()}
                                        </Typography>
                                    </Box>

                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Risk Amount:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600, color: theme.palette.error.main }}>
                                            ${result.risk_amount.toLocaleString()} ({result.risk_percent}%)
                                        </Typography>
                                    </Box>

                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            Risk Per Share:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            ${result.risk_per_share.toFixed(2)}
                                        </Typography>
                                    </Box>
                                </Box>
                            </Box>

                            <Box
                                sx={{
                                    backgroundColor: alpha(theme.palette.success.main, 0.1),
                                    borderRadius: 2,
                                    p: 2,
                                }}
                            >
                                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, color: theme.palette.success.main }}>
                                    Target Prices (Risk:Reward)
                                </Typography>

                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            1:1 Target:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            ${result.reward_ratios['1:1'].toLocaleString()}
                                        </Typography>
                                    </Box>

                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            1:2 Target:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            ${result.reward_ratios['1:2'].toLocaleString()}
                                        </Typography>
                                    </Box>

                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <Typography variant="body2" color="text.secondary">
                                            1:3 Target:
                                        </Typography>
                                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                            ${result.reward_ratios['1:3'].toLocaleString()}
                                        </Typography>
                                    </Box>
                                </Box>
                            </Box>
                        </>
                    )}
                </Box>
            </DialogContent>

            <DialogActions>
                <Button onClick={handleClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
}
