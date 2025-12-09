import { useState, useEffect } from 'react';
import { Box, Typography, Tooltip, Popover, IconButton } from '@mui/material';
import { Wifi as WifiIcon, WifiOff as WifiOffIcon, Speed as SpeedIcon, Info as InfoIcon } from '@mui/icons-material';
import apiService from '../services/api';
import TradingReadiness from './TradingReadiness';

export default function ConnectionStatus() {
    const [status, setStatus] = useState<any>(null);
    const [connected, setConnected] = useState(false);
    const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

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

    useEffect(() => {
        checkStatus();
        const interval = setInterval(checkStatus, 10000); // Check every 10s
        return () => clearInterval(interval);
    }, []);

    const handleClick = (event: React.MouseEvent<HTMLElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const open = Boolean(anchorEl);

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
        <>
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2,
                    cursor: 'pointer',
                    '&:hover': { opacity: 0.8 }
                }}
                onClick={handleClick}
            >
                <Tooltip title={`Binance API Latency: ${status?.latency?.binance_api || 0}ms`}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
                        <SpeedIcon fontSize="small" />
                        <Typography variant="caption">{status?.latency?.binance_api || 0}ms</Typography>
                    </Box>
                </Tooltip>

                <Tooltip title="Click for System Readiness Details">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: connected ? 'success.main' : 'warning.main' }}>
                        <WifiIcon fontSize="small" />
                        <Typography variant="caption" sx={{ fontWeight: 600 }}>
                            {connected ? 'Online' : 'Degraded'}
                        </Typography>
                    </Box>
                </Tooltip>
            </Box>

            <Popover
                open={open}
                anchorEl={anchorEl}
                onClose={handleClose}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'right',
                }}
                transformOrigin={{
                    vertical: 'top',
                    horizontal: 'right',
                }}
            >
                <TradingReadiness data={status?.readiness} />
            </Popover>
        </>
    );
}
