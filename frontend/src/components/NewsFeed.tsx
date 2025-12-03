import { useState, useEffect } from 'react';
import { Box, Typography, Paper, List, ListItem, Chip, Link, Divider } from '@mui/material';
import { AccessTime } from '@mui/icons-material';
import apiService from '../services/api';

export default function NewsFeed() {
    const [news, setNews] = useState<any[]>([]);

    useEffect(() => {
        const fetchNews = async () => {
            try {
                const result = await apiService.getNews();
                setNews(result);
            } catch (error) {
                console.error('Error fetching news:', error);
            }
        };
        fetchNews();
    }, []);

    const getSentimentColor = (sentiment: string) => {
        switch (sentiment) {
            case 'Positive': return 'success';
            case 'Negative': return 'error';
            default: return 'default';
        }
    };

    return (
        <Paper sx={{ height: '100%', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
                <Typography variant="h6">Crypto News</Typography>
            </Box>
            <List sx={{ overflow: 'auto', flexGrow: 1, p: 0 }}>
                {news.map((item, index) => (
                    <Box key={item.id}>
                        <ListItem alignItems="flex-start" sx={{ flexDirection: 'column', gap: 1 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                                <Chip
                                    label={item.source}
                                    size="small"
                                    variant="outlined"
                                    sx={{ fontSize: '0.7rem', height: 20 }}
                                />
                                <Chip
                                    label={item.sentiment}
                                    color={getSentimentColor(item.sentiment) as any}
                                    size="small"
                                    sx={{ fontSize: '0.7rem', height: 20 }}
                                />
                            </Box>
                            <Link href={item.url} target="_blank" color="inherit" underline="hover">
                                <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
                                    {item.title}
                                </Typography>
                            </Link>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
                                <AccessTime sx={{ fontSize: 12 }} />
                                <Typography variant="caption">
                                    {new Date(item.published_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </Typography>
                            </Box>
                        </ListItem>
                        {index < news.length - 1 && <Divider component="li" />}
                    </Box>
                ))}
            </List>
        </Paper>
    );
}
