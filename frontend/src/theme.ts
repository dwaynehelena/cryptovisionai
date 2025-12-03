import { createTheme } from '@mui/material/styles';
import type { ThemeOptions } from '@mui/material/styles';

// Material Design 3 Expressive Theme
const md3ExpressiveTheme: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: {
      main: '#A8C5FF', // MD3 Primary
      light: '#D3E3FD',
      dark: '#4285F4',
      contrastText: '#001A41',
    },
    secondary: {
      main: '#BCC7DC', // MD3 Secondary
      light: '#E1E8F5',
      dark: '#5B6B7F',
      contrastText: '#0E1E30',
    },
    background: {
      default: '#0E1419', // Deep dark background
      paper: '#1A1F26', // Elevated surface
    },
    error: {
      main: '#FFB4AB',
      dark: '#93000A',
    },
    success: {
      main: '#79DDA8',
      dark: '#006E2C',
    },
    warning: {
      main: '#EFB8C8',
      dark: '#93072E',
    },
    info: {
      main: '#A8C5FF',
      dark: '#003C8F',
    },
    text: {
      primary: '#E3E3E8',
      secondary: '#C6C6CC',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 600,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 16, // MD3 uses larger radius
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#1A1F26',
          borderRadius: 20,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
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
          backgroundColor: '#1A1F26',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: '#1A1F26',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.3)',
        },
      },
    },
  },
};

export const theme = createTheme(md3ExpressiveTheme);
