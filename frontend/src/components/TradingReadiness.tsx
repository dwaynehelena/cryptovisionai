import React from 'react';
import {
    Box,
    Typography,
    Paper,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Chip,
    Divider,
    CircularProgress
} from '@mui/material';
import {
    CheckCircle as CheckCircleIcon,
    Error as ErrorIcon,
    Warning as WarningIcon,
    Help as HelpIcon
} from '@mui/icons-material';

interface ReadinessComponent {
    status: string;
    details: string;
}

interface ReadinessData {
    ready: boolean;
    components: {
        [key: string]: ReadinessComponent;
    };
    timestamp: number;
}

interface TradingReadinessProps {
    data: ReadinessData | null;
    loading?: boolean;
}

const TradingReadiness: React.FC<TradingReadinessProps> = ({ data, loading }) => {
    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} />
            </Box>
        );
    }

    if (!data) {
        return (
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography color="text.secondary">No readiness data available</Typography>
            </Box>
        );
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'operational':
                return <CheckCircleIcon color="success" />;
            case 'error':
            case 'not_ready':
            case 'restricted':
                return <ErrorIcon color="error" />;
            case 'warning':
                return <WarningIcon color="warning" />;
            default:
                return <HelpIcon color="disabled" />;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'operational':
                return 'success';
            case 'error':
            case 'not_ready':
            case 'restricted':
                return 'error';
            case 'warning':
                return 'warning';
            default:
                return 'default';
        }
    };

    const formatComponentName = (name: string) => {
        return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    };

    return (
        <Paper sx={{ p: 2, minWidth: 300 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">System Readiness</Typography>
                <Chip
                    label={data.ready ? "READY TO TRADE" : "NOT READY"}
                    color={data.ready ? "success" : "error"}
                    variant="filled"
                    size="small"
                />
            </Box>
            <Divider />
            <List dense>
                {Object.entries(data.components).map(([key, component]) => (
                    <ListItem key={key}>
                        <ListItemIcon>
                            {getStatusIcon(component.status)}
                        </ListItemIcon>
                        <ListItemText
                            primary={formatComponentName(key)}
                            secondary={component.details}
                            primaryTypographyProps={{ fontWeight: 500 }}
                        />
                    </ListItem>
                ))}
            </List>
        </Paper>
    );
};

export default TradingReadiness;
