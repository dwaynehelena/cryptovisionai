
import React, { useEffect, useState } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    LinearProgress,
    List,
    ListItem,
    ListItemText,
    Chip,
    Divider,
    useTheme,
    Tooltip
} from '@mui/material';
import { apiService } from '../services/api';
import PsychologyIcon from '@mui/icons-material/Psychology';

interface ModelInfo {
    name: string;
    accuracy: number;
    type: string;
    weight?: number;
}

interface ModelPerformanceData {
    ensemble_type: string;
    last_trained: string;
    models: ModelInfo[];
    overall_accuracy: number;
}

const ModelPerformanceCard: React.FC = () => {
    const theme = useTheme();
    const [data, setData] = useState<ModelPerformanceData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await apiService.getModelPerformance();
                setData(response);
            } catch (error) {
                console.error("Failed to fetch model performance:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 60000); // Refresh every minute
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Card sx={{ height: '100%' }}>
                <CardContent>
                    <Typography gutterBottom variant="h6" component="div">
                        <Box display="flex" alignItems="center" gap={1}>
                            <PsychologyIcon color="primary" />
                            Model Performance
                        </Box>
                    </Typography>
                    <LinearProgress />
                </CardContent>
            </Card>
        );
    }

    if (!data) return null;

    return (
        <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent>
                <Typography gutterBottom variant="h6" component="div">
                    <Box display="flex" alignItems="center" gap={1}>
                        <PsychologyIcon color="primary" />
                        AI Ensemble Performance
                    </Box>
                </Typography>

                <Box sx={{ mb: 3, textAlign: 'center' }}>
                    <Typography variant="h3" color="primary" fontWeight="bold">
                        {data.overall_accuracy}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Overall Accuracy
                    </Typography>
                    <Chip
                        label={`${data.ensemble_type} Ensemble`}
                        size="small"
                        color="secondary"
                        variant="outlined"
                        sx={{ mt: 1 }}
                    />
                </Box>

                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Individual Models
                </Typography>
                <Divider sx={{ mb: 1 }} />

                <List disablePadding>
                    {data.models.map((model, index) => (
                        <ListItem key={index} disablePadding sx={{ mb: 1.5, display: 'block' }}>
                            <Box display="flex" justifyContent="space-between" mb={0.5}>
                                <Typography variant="body2" fontWeight="medium">
                                    {model.name}
                                </Typography>
                                <Typography variant="body2" color={
                                    model.accuracy > 85 ? 'success.main' :
                                        model.accuracy > 75 ? 'warning.main' : 'error.main'
                                }>
                                    {model.accuracy}%
                                </Typography>
                            </Box>
                            <Tooltip title={`Accuracy: ${model.accuracy}%`}>
                                <LinearProgress
                                    variant="determinate"
                                    value={model.accuracy}
                                    color={
                                        model.accuracy > 85 ? 'success' :
                                            model.accuracy > 75 ? 'warning' : 'error'
                                    }
                                    sx={{ height: 6, borderRadius: 3 }}
                                />
                            </Tooltip>
                        </ListItem>
                    ))}
                </List>

                <Box mt={2}>
                    <Typography variant="caption" color="text.secondary" display="block">
                        Last Trained: {new Date(data.last_trained).toLocaleString()}
                    </Typography>
                </Box>
            </CardContent>
        </Card>
    );
};

export default ModelPerformanceCard;
