import { createTheme } from '@mui/material/styles';
import { createContext, useMemo, useState } from 'react';
import type { ThemeOptions, PaletteMode } from '@mui/material';

// Context for color mode
export const ColorModeContext = createContext({ toggleColorMode: () => { } });

// Hook to manage theme state
export const useColorMode = () => {
  const [mode, setMode] = useState<PaletteMode>(() => {
    const savedMode = localStorage.getItem('colorMode');
    return (savedMode as PaletteMode) || 'dark';
  });

  const colorMode = useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => {
          const newMode = prevMode === 'light' ? 'dark' : 'light';
          localStorage.setItem('colorMode', newMode);
          return newMode;
        });
      },
    }),
    [],
  );

  const theme = useMemo(() => {
    const getDesignTokens = (mode: PaletteMode): ThemeOptions => ({
      palette: {
        mode,
        ...(mode === 'dark'
          ? {
            // Dark Mode Palette
            primary: {
              main: '#A8C5FF',
              light: '#D3E3FD',
              dark: '#4285F4',
              contrastText: '#001A41',
            },
            secondary: {
              main: '#BCC7DC',
              light: '#E1E8F5',
              dark: '#5B6B7F',
              contrastText: '#0E1E30',
            },
            background: {
              default: '#0E1419',
              paper: '#1A1F26',
            },
            text: {
              primary: '#E3E3E8',
              secondary: '#C6C6CC',
            },
          }
          : {
            // Light Mode Palette
            primary: {
              main: '#4285F4',
              light: '#D3E3FD',
              dark: '#0B57D0',
              contrastText: '#FFFFFF',
            },
            secondary: {
              main: '#5B6B7F',
              light: '#D3E3FD',
              dark: '#2A3C50',
              contrastText: '#FFFFFF',
            },
            background: {
              default: '#F8F9FA',
              paper: '#FFFFFF',
            },
            text: {
              primary: '#1A1F26',
              secondary: '#444746',
            },
          }),
      },
      typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: { fontSize: '2.5rem', fontWeight: 700, letterSpacing: '-0.02em' },
        h2: { fontSize: '2rem', fontWeight: 600, letterSpacing: '-0.01em' },
        h3: { fontSize: '1.75rem', fontWeight: 600 },
        h4: { fontSize: '1.5rem', fontWeight: 600 },
        h5: { fontSize: '1.25rem', fontWeight: 600 },
        h6: { fontSize: '1.125rem', fontWeight: 600 },
        body1: { fontSize: '1rem', lineHeight: 1.6 },
        body2: { fontSize: '0.875rem', lineHeight: 1.6 },
      },
      shape: {
        borderRadius: 16,
      },
      components: {
        MuiCard: {
          styleOverrides: {
            root: {
              backgroundImage: 'none',
              borderRadius: 20,
              boxShadow: mode === 'dark'
                ? '0 4px 20px rgba(0, 0, 0, 0.4)'
                : '0 2px 12px rgba(0, 0, 0, 0.08)',
              backgroundColor: mode === 'dark' ? '#1A1F26' : '#FFFFFF',
            },
          },
        },
        MuiButton: {
          styleOverrides: {
            root: {
              borderRadius: 20,
              textTransform: 'none',
              fontWeight: 600,
              padding: '10px 24px',
            },
            contained: {
              boxShadow: 'none',
              '&:hover': {
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
              },
            },
          },
        },
        MuiPaper: {
          styleOverrides: {
            root: {
              backgroundImage: 'none',
              backgroundColor: mode === 'dark' ? '#1A1F26' : '#FFFFFF',
            },
          },
        },
        MuiAppBar: {
          styleOverrides: {
            root: {
              backgroundImage: 'none',
              backgroundColor: mode === 'dark' ? '#1A1F26' : '#FFFFFF',
              color: mode === 'dark' ? '#E3E3E8' : '#1A1F26',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            },
          },
        },
      },
    });

    return createTheme(getDesignTokens(mode));
  }, [mode]);

  return { theme, colorMode, mode };
};
