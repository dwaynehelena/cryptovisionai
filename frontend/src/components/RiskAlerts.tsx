import React, { useEffect, useState } from 'react';
import { Snackbar, Alert } from '@mui/material';
import { apiService as api } from '../services/api';

interface Violation {
    timestamp: string;
    type: string;
    message: string;
}

const RiskAlerts: React.FC = () => {
    const [open, setOpen] = useState(false);
    const [currentViolation, setCurrentViolation] = useState<Violation | null>(null);
    const [lastViolationTime, setLastViolationTime] = useState<string | null>(null);

    useEffect(() => {
        const checkViolations = async () => {
            try {
                const data = await api.getRiskViolations(1);
                if (data.violations && data.violations.length > 0) {
                    const latest = data.violations[0];

                    // Show alert if it's new (different timestamp than last seen)
                    if (latest.timestamp !== lastViolationTime) {
                        setCurrentViolation(latest);
                        setLastViolationTime(latest.timestamp);
                        setOpen(true);
                    }
                }
            } catch (error) {
                console.error('Failed to check risk violations:', error);
            }
        };

        const interval = setInterval(checkViolations, 5000); // Check every 5 seconds
        return () => clearInterval(interval);
    }, [lastViolationTime]);

    const handleClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
        if (reason === 'clickaway') {
            return;
        }
        setOpen(false);
    };

    if (!currentViolation) return null;

    return (
        <Snackbar
            open={open}
            autoHideDuration={6000}
            onClose={handleClose}
            anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
            <Alert onClose={handleClose} severity="error" sx={{ width: '100%' }} variant="filled">
                {currentViolation.message}
            </Alert>
        </Snackbar>
    );
};

export default RiskAlerts;
