import React, { useEffect, useState } from 'react';
import {
    Paper,
    Typography,
    Box,
    Grid,
    Slider,
    Switch,
    FormControlLabel,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogContentText,
    DialogActions,
    Alert,
    Divider
} from '@mui/material';
import {
    Settings as SettingsIcon,
    Save as SaveIcon,
    Warning as WarningIcon
} from '@mui/icons-material';
import { apiService as api } from '../services/api';

interface RiskLimits {
    max_position_size_pct: number;
    max_risk_per_trade_pct: number;
    max_drawdown_pct: number;
    daily_loss_limit_pct: number;
    auto_stop_loss_enabled: boolean;
    atr_multiplier: number;
    max_correlation: number;
    trading_enabled: boolean;
}

const RiskSettings: React.FC = () => {
    const [limits, setLimits] = useState<RiskLimits | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [saving, setSaving] = useState<boolean>(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
    const [emergencyOpen, setEmergencyOpen] = useState(false);

    useEffect(() => {
        fetchLimits();
    }, []);

    const fetchLimits = async () => {
        try {
            setLoading(true);
            const data = await api.getRiskLimits();
            setLimits(data);
        } catch (err) {
            console.error('Failed to fetch risk limits:', err);
            setMessage({ type: 'error', text: 'Failed to load risk settings' });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!limits) return;

        try {
            setSaving(true);
            await api.updateRiskLimits(limits);
            setMessage({ type: 'success', text: 'Risk settings saved successfully' });
            setTimeout(() => setMessage(null), 3000);
        } catch (err) {
            console.error('Failed to save risk limits:', err);
            setMessage({ type: 'error', text: 'Failed to save settings' });
        } finally {
            setSaving(false);
        }
    };

    const handleEmergencyClose = async () => {
        try {
            await api.emergencyCloseAll();
            setEmergencyOpen(false);
            setMessage({ type: 'success', text: 'Emergency close executed successfully' });
            fetchLimits(); // Refresh to see trading disabled status
        } catch (err) {
            console.error('Emergency close failed:', err);
            setMessage({ type: 'error', text: 'Emergency close failed' });
        }
    };

    const handleToggleTrading = async () => {
        if (!limits) return;
        try {
            if (limits.trading_enabled) {
                await api.disableTrading();
            } else {
                await api.enableTrading();
            }
            fetchLimits();
        } catch (err) {
            console.error('Failed to toggle trading:', err);
        }
    };

    if (loading || !limits) return null;

    return (
        <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h6" display="flex" alignItems="center">
                    <SettingsIcon sx={{ mr: 1 }} />
                    Risk Configuration
                </Typography>
                <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={handleSave}
                    disabled={saving}
                >
                    Save Changes
                </Button>
            </Box>

            {message && (
                <Alert severity={message.type} sx={{ mb: 3 }}>
                    {message.text}
                </Alert>
            )}

            <Grid container spacing={4}>
                <Grid item xs={12} md={6}>
                    <Typography gutterBottom>Max Position Size (% of Portfolio)</Typography>
                    <Slider
                        value={limits.max_position_size_pct}
                        onChange={(_, val) => setLimits({ ...limits, max_position_size_pct: val as number })}
                        valueLabelDisplay="auto"
                        step={1}
                        marks
                        min={1}
                        max={50}
                    />

                    <Typography gutterBottom sx={{ mt: 2 }}>Max Risk Per Trade (%)</Typography>
                    <Slider
                        value={limits.max_risk_per_trade_pct}
                        onChange={(_, val) => setLimits({ ...limits, max_risk_per_trade_pct: val as number })}
                        valueLabelDisplay="auto"
                        step={0.1}
                        marks
                        min={0.1}
                        max={5}
                    />

                    <Typography gutterBottom sx={{ mt: 2 }}>Max Drawdown Limit (%)</Typography>
                    <Slider
                        value={limits.max_drawdown_pct}
                        onChange={(_, val) => setLimits({ ...limits, max_drawdown_pct: val as number })}
                        valueLabelDisplay="auto"
                        step={1}
                        marks
                        min={5}
                        max={50}
                    />
                </Grid>

                <Grid item xs={12} md={6}>
                    <Typography variant="subtitle1" gutterBottom>Automated Controls</Typography>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={limits.auto_stop_loss_enabled}
                                onChange={(e) => setLimits({ ...limits, auto_stop_loss_enabled: e.target.checked })}
                            />
                        }
                        label="Enable Auto Stop-Loss"
                    />

                    <Box mt={2}>
                        <Typography gutterBottom>ATR Multiplier (for Stop-Loss)</Typography>
                        <Slider
                            value={limits.atr_multiplier}
                            onChange={(_, val) => setLimits({ ...limits, atr_multiplier: val as number })}
                            valueLabelDisplay="auto"
                            step={0.1}
                            marks
                            min={1}
                            max={5}
                            disabled={!limits.auto_stop_loss_enabled}
                        />
                    </Box>

                    <Divider sx={{ my: 3 }} />

                    <Typography variant="subtitle1" color="error" gutterBottom>
                        Danger Zone
                    </Typography>

                    <Box display="flex" gap={2}>
                        <Button
                            variant="outlined"
                            color={limits.trading_enabled ? "error" : "success"}
                            onClick={handleToggleTrading}
                        >
                            {limits.trading_enabled ? "Disable Trading" : "Enable Trading"}
                        </Button>

                        <Button
                            variant="contained"
                            color="error"
                            startIcon={<WarningIcon />}
                            onClick={() => setEmergencyOpen(true)}
                        >
                            Emergency Close All
                        </Button>
                    </Box>
                </Grid>
            </Grid>

            {/* Emergency Confirmation Dialog */}
            <Dialog
                open={emergencyOpen}
                onClose={() => setEmergencyOpen(false)}
            >
                <DialogTitle color="error" display="flex" alignItems="center">
                    <WarningIcon sx={{ mr: 1 }} />
                    Emergency Action
                </DialogTitle>
                <DialogContent>
                    <DialogContentText>
                        Are you sure you want to close ALL open positions immediately?
                        This action cannot be undone and will execute market orders for all positions.
                        Trading will be disabled after this action.
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setEmergencyOpen(false)}>Cancel</Button>
                    <Button onClick={handleEmergencyClose} color="error" variant="contained" autoFocus>
                        Confirm Close All
                    </Button>
                </DialogActions>
            </Dialog>
        </Paper>
    );
};

export default RiskSettings;
