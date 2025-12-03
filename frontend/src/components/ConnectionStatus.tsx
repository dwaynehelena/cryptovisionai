import { useState, useEffect } from 'react';
import { Box, Typography, Tooltip } from '@mui/material';
import { Wifi as WifiIcon, WifiOff as WifiOffIcon, Speed as SpeedIcon } from '@mui/icons-material';
import apiService from '../services/api';

export default function ConnectionStatus() {
    const [status, setStatus] = useState<any>(null);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        const checkStatus = async () => {
            try {
                const data = await apiService.getSystemStatus();
                setStatus(data);
                setConnected(data.status === 'operational');
            } catch (error) {
                setConnected(false);
                setStatus(null);
            }
        };

        checkStatus();
        const interval = setInterval(checkStatus, 10000); // Check every 10s
        return () => clearInterval(interval);
    }, []);

    if (!status && !connected) {
        return (
            <Tooltip title="Disconnected from Server">
                <Box sx={{ display: 'flex', alignItems: 'center', color: 'error.main', gap: 1 }}>
                    <WifiOffIcon fontSize="small" />
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>Offline</Typography>
                </Box>
            </Tooltip>
        );
    }

    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Tooltip title={`Binance API Latency: ${status?.latency?.binance_api || 0}ms`}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
                    <SpeedIcon fontSize="small" />
                    <Typography variant="caption">{status?.latency?.binance_api || 0}ms</Typography>
                </Box>
            </Tooltip>

            <Tooltip title={`System Status: ${status?.status?.toUpperCase()}`}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: connected ? 'success.main' : 'warning.main' }}>
                    <WifiIcon fontSize="small" />
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                        {connected ? 'Online' : 'Degraded'}
                    </Typography>
                </Box>
            </Tooltip>
        </Box>
    );
}
