import { useState } from 'react';
import {
    Box,
    Typography,
    Paper,
    Tabs,
    Tab,
    TextField,
    Button,
    Switch,
    FormControlLabel,
    Alert,
    CircularProgress,
    Grid
} from '@mui/material';
import {
    VpnKey as KeyIcon,
    Settings as SettingsIcon,
    Security as SecurityIcon,
    Notifications as NotificationsIcon
} from '@mui/icons-material';
import apiService from '../services/api';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`settings-tabpanel-${index}`}
            aria-labelledby={`settings-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ p: 3 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

export default function Settings() {
    const [value, setValue] = useState(0);
    const [apiKey, setApiKey] = useState('');
    const [apiSecret, setApiSecret] = useState('');
    const [testnet, setTestnet] = useState(true);
    const [validating, setValidating] = useState(false);
    const [validationResult, setValidationResult] = useState<{ valid: boolean; message: string } | null>(null);

    const handleChange = (_event: React.SyntheticEvent, newValue: number) => {
        setValue(newValue);
    };

    const handleValidateKeys = async () => {
        setValidating(true);
        setValidationResult(null);
        try {
            const result = await apiService.validateApiKeys(apiKey, apiSecret, testnet);

            if (result.valid) {
                try {
                    await apiService.updateSystemConfig(apiKey, apiSecret, testnet);
                    setValidationResult({ valid: true, message: 'Keys validated and system updated successfully!' });
                } catch (updateError) {
                    setValidationResult({ valid: false, message: 'Keys valid but failed to update system configuration.' });
                }
            } else {
                setValidationResult(result);
            }
        } catch (error) {
            setValidationResult({ valid: false, message: 'Network error or server unavailable' });
        } finally {
            setValidating(false);
        }
    };

    return (
        <Paper sx={{ width: '100%', borderRadius: 2, overflow: 'hidden' }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={value} onChange={handleChange} aria-label="settings tabs">
                    <Tab icon={<SettingsIcon />} label="General" iconPosition="start" />
                    <Tab icon={<KeyIcon />} label="API Keys" iconPosition="start" />
                    <Tab icon={<SecurityIcon />} label="Risk" iconPosition="start" />
                    <Tab icon={<NotificationsIcon />} label="Notifications" iconPosition="start" />
                </Tabs>
            </Box>

            {/* General Settings */}
            <TabPanel value={value} index={0}>
                <Typography variant="h6" gutterBottom>General Preferences</Typography>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <FormControlLabel
                            control={<Switch defaultChecked />}
                            label="Auto-connect WebSocket on startup"
                        />
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <FormControlLabel
                            control={<Switch />}
                            label="Compact Mode"
                        />
                    </Grid>
                </Grid>
            </TabPanel>

            {/* API Keys */}
            <TabPanel value={value} index={1}>
                <Typography variant="h6" gutterBottom>Binance API Configuration</Typography>
                <Alert severity="info" sx={{ mb: 3 }}>
                    API keys are validated against Binance but are NOT stored on the server for security.
                    They are used for the current session or stored locally in your browser if configured.
                </Alert>

                <Box component="form" noValidate autoComplete="off">
                    <Grid container spacing={3}>
                        <Grid item xs={12}>
                            <FormControlLabel
                                control={<Switch checked={testnet} onChange={(e) => setTestnet(e.target.checked)} />}
                                label="Use Testnet"
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                label="API Key"
                                value={apiKey}
                                onChange={(e) => setApiKey(e.target.value)}
                                type="password"
                            />
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                label="API Secret"
                                value={apiSecret}
                                onChange={(e) => setApiSecret(e.target.value)}
                                type="password"
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Button
                                variant="contained"
                                onClick={handleValidateKeys}
                                disabled={validating || !apiKey || !apiSecret}
                                startIcon={validating && <CircularProgress size={20} />}
                            >
                                {validating ? 'Validating...' : 'Validate & Save'}
                            </Button>
                        </Grid>
                    </Grid>
                </Box>

                {validationResult && (
                    <Alert severity={validationResult.valid ? "success" : "error"} sx={{ mt: 3 }}>
                        {validationResult.message}
                    </Alert>
                )}
            </TabPanel>

            {/* Risk Settings */}
            <TabPanel value={value} index={2}>
                <Typography variant="h6" gutterBottom>Global Risk Parameters</Typography>
                <Alert severity="warning" sx={{ mb: 3 }}>
                    These settings act as a hard stop for all trading activities.
                </Alert>
                <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Max Daily Drawdown (%)"
                            defaultValue="5.0"
                            type="number"
                        />
                    </Grid>
                    <Grid item xs={12} md={6}>
                        <TextField
                            fullWidth
                            label="Max Position Size (% of Equity)"
                            defaultValue="10.0"
                            type="number"
                        />
                    </Grid>
                    <Grid item xs={12}>
                        <Button variant="contained" color="primary">Update Risk Limits</Button>
                    </Grid>
                </Grid>
            </TabPanel>

            {/* Notifications */}
            <TabPanel value={value} index={3}>
                <Typography variant="h6" gutterBottom>Alert Preferences</Typography>
                <Grid container spacing={2}>
                    <Grid item xs={12}>
                        <FormControlLabel control={<Switch defaultChecked />} label="Trade Executions" />
                    </Grid>
                    <Grid item xs={12}>
                        <FormControlLabel control={<Switch defaultChecked />} label="Risk Limit Warnings" />
                    </Grid>
                    <Grid item xs={12}>
                        <FormControlLabel control={<Switch />} label="Price Alerts" />
                    </Grid>
                </Grid>
            </TabPanel>
        </Paper>
    );
}
