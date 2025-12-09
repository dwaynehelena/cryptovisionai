import { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Box,
    useTheme,
} from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import apiService from '../services/api';

interface Allocation {
    asset: string;
    value: number;
    percentage: number;
    quantity: number;
}

interface AllocationData {
    total_value: number;
    allocations: Allocation[];
    asset_count: number;
}

export default function PortfolioAllocation() {
    const theme = useTheme();
    const [data, setData] = useState<AllocationData | null>(null);

    useEffect(() => {
        fetchAllocation();
        const interval = setInterval(fetchAllocation, 10000); // Update every 10s
        return () => clearInterval(interval);
    }, []);

    const fetchAllocation = async () => {
        try {
            const allocationData = await apiService.getPortfolioAllocation();
            setData(allocationData);
        } catch (error) {
            console.error('Error fetching allocation:', error);
        }
    };

    if (!data) {
        return null;
    }

    // Colors for pie chart
    const COLORS = [
        theme.palette.primary.main,
        theme.palette.secondary.main,
        theme.palette.success.main,
        theme.palette.warning.main,
        theme.palette.error.main,
        theme.palette.info.main,
        '#FF6B9D',
        '#C9CBFF',
        '#FFE66D',
        '#95E1D3',
    ];

    const chartData = data.allocations.map((allocation) => ({
        name: allocation.asset,
        value: allocation.percentage,
        actualValue: allocation.value,
    }));

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            return (
                <Box
                    sx={{
                        backgroundColor: theme.palette.background.paper,
                        border: `1px solid ${theme.palette.divider}`,
                        borderRadius: 1,
                        p: 1.5,
                    }}
                >
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {payload[0].name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        {payload[0].value.toFixed(2)}% (${payload[0].payload.actualValue.toLocaleString()})
                    </Typography>
                </Box>
            );
        }
        return null;
    };

    return (
        <Card>
            <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 700, mb: 2 }}>
                    Portfolio Allocation
                </Typography>

                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                                outerRadius={100}
                                fill="#8884d8"
                                dataKey="value"
                            >
                                {chartData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip />} />
                        </PieChart>
                    </ResponsiveContainer>

                    <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                            Holdings ({data.asset_count} assets)
                        </Typography>
                        {data.allocations.map((allocation, index) => (
                            <Box
                                key={allocation.asset}
                                sx={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    p: 1,
                                    borderRadius: 1,
                                    '&:hover': {
                                        backgroundColor: theme.palette.action.hover,
                                    },
                                }}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    <Box
                                        sx={{
                                            width: 12,
                                            height: 12,
                                            borderRadius: '50%',
                                            backgroundColor: COLORS[index % COLORS.length],
                                        }}
                                    />
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                        {allocation.asset}
                                    </Typography>
                                </Box>
                                <Box sx={{ textAlign: 'right' }}>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                        ${allocation.value.toLocaleString()}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {allocation.percentage.toFixed(2)}%
                                    </Typography>
                                </Box>
                            </Box>
                        ))}
                    </Box>
                </Box>
            </CardContent>
        </Card>
    );
}
