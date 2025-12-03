import { useHotkeys } from 'react-hotkeys-hook';
import { useColorMode } from '../theme';
import { useTheme } from '@mui/material/styles';
import { Box, Typography, Modal, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import { useState } from 'react';

interface Props {
    onFocusBuy: () => void;
    onFocusSell: () => void;
}

export default function KeyboardShortcuts({ onFocusBuy, onFocusSell }: Props) {
    const { colorMode } = useColorMode();
    const { toggleColorMode } = colorMode;
    const [open, setOpen] = useState(false);
    const theme = useTheme();

    // Toggle Theme: Shift + D
    useHotkeys('shift+d', () => toggleColorMode());

    // Focus Buy Order: Shift + B
    useHotkeys('shift+b', (e) => {
        e.preventDefault();
        onFocusBuy();
    });

    // Focus Sell Order: Shift + S
    useHotkeys('shift+s', (e) => {
        e.preventDefault();
        onFocusSell();
    });

    // Show Shortcuts Help: Shift + ?
    useHotkeys('shift+/', () => setOpen(true));

    return (
        <Modal
            open={open}
            onClose={() => setOpen(false)}
            aria-labelledby="keyboard-shortcuts-title"
        >
            <Box sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: 400,
                bgcolor: 'background.paper',
                border: '2px solid #000',
                boxShadow: 24,
                p: 4,
                borderRadius: 2,
            }}>
                <Typography id="keyboard-shortcuts-title" variant="h6" component="h2" mb={2}>
                    Keyboard Shortcuts
                </Typography>
                <TableContainer component={Paper}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Action</TableCell>
                                <TableCell align="right">Shortcut</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                <TableCell>Toggle Theme</TableCell>
                                <TableCell align="right"><Box component="span" sx={{ bgcolor: theme.palette.action.selected, px: 1, borderRadius: 1 }}>Shift + D</Box></TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>Focus Buy Order</TableCell>
                                <TableCell align="right"><Box component="span" sx={{ bgcolor: theme.palette.action.selected, px: 1, borderRadius: 1 }}>Shift + B</Box></TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>Focus Sell Order</TableCell>
                                <TableCell align="right"><Box component="span" sx={{ bgcolor: theme.palette.action.selected, px: 1, borderRadius: 1 }}>Shift + S</Box></TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell>Show Shortcuts</TableCell>
                                <TableCell align="right"><Box component="span" sx={{ bgcolor: theme.palette.action.selected, px: 1, borderRadius: 1 }}>Shift + ?</Box></TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </Box>
        </Modal>
    );
}
