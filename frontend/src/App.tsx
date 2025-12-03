import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useColorMode, ColorModeContext } from './theme';
import Dashboard from './components/Dashboard';

function App() {
  const { theme, colorMode } = useColorMode();

  return (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Dashboard />
      </ThemeProvider>
    </ColorModeContext.Provider>
  );
}

export default App;
