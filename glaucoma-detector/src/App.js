import React, { useState } from 'react';
import './App.css';
import {
  Container,
  Typography,
  Button,
  Box,
  CircularProgress,
  LinearProgress,
  Card,
  CardContent,
  TextareaAutosize,
  AppBar,
  Toolbar,
  Snackbar,
  Tab,
  Tabs,
} from '@mui/material';
import { styled } from '@mui/system';
import { Routes, Route, Link } from 'react-router-dom';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import AssessmentIcon from '@mui/icons-material/Assessment';

const Input = styled('input')({
  display: 'none',
});

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState('');
  const [oldResults, setOldResults] = useState([]);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);

  const backendURL = 'http://127.0.0.1:5000/api/calculate_vcdr';

  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  // Handle file selection
  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);

    if (files.length === 0) {
      setSnackbarMessage('No files selected. Please select valid images.');
      setSnackbarOpen(true);
      return;
    }

    setSelectedFiles(files);
    setResults('');
    setProgress(0);
    setLoading(false);
    setUploadSuccess(false);
    setSnackbarMessage('Files selected successfully.');
    setSnackbarOpen(true);
  };

  // Handle classification submission
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (selectedFiles.length > 0) {
      setLoading(true); // Start loading

      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append('images', file);
      });

      try {
        const response = await fetch(backendURL, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Server response:', data);
          // Format the results for display
          const formattedResults = data.results
            .map(
              (result) =>
                `File: ${result.file}\n  vCDR: ${result.vCDR}\n  Prediction: ${result.prediction}\n`
            )
            .join('\n');

          setResults(formattedResults); // Update results
          setSnackbarMessage('Classification completed successfully!');
          setSnackbarOpen(true);
          setOldResults((prev) => [
            ...prev,
            { timestamp: new Date(), results: formattedResults },
          ]); // Save results to archive
        } else {
          const errorData = await response.json();
          console.error('Error during classification:', errorData.error);
          setSnackbarMessage('Error during classification: ' + errorData.error);
          setSnackbarOpen(true);
        }
      } catch (error) {
        console.error('Error during classification:', error);
        setSnackbarMessage('Error during classification: ' + error.message);
        setSnackbarOpen(true);
      } finally {
        setLoading(false); // Stop loading spinner
      }
    } else {
      setSnackbarMessage('Please select files to upload.');
      setSnackbarOpen(true);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  return (
    <>
      <AppBar position="static" style={{ backgroundColor: darkMode ? '#1a237e' : '#1976d2' }}>
        <Toolbar>
          <Typography variant="h6" style={{ flexGrow: 1 }}>
            Glaucoma OCT Detection
          </Typography>
          <Button color="inherit" onClick={toggleDarkMode}>
            {darkMode ? 'Light Mode' : 'Dark Mode'}
          </Button>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          <Button color="inherit" component={Link} to="/dashboard">
            Dashboard
          </Button>
        </Toolbar>
      </AppBar>

      <Routes>
        <Route
          path="/"
          element={
            <Container maxWidth="md">
              <Card
                style={{
                  borderRadius: '15px',
                  padding: '2rem',
                  backgroundColor: darkMode ? '#283593' : '#e3f2fd',
                }}
              >
                <CardContent>
                  <Typography variant="h4" gutterBottom>
                    Upload and Classify Images
                  </Typography>
                  <Box display="flex" justifyContent="center" gap="1rem" marginBottom="1rem">
                    <label htmlFor="file-upload">
                      <Input accept="image/*" id="file-upload" type="file" multiple onChange={handleFileChange} />
                      <Button
                        variant="contained"
                        component="span"
                        startIcon={<CloudUploadIcon />}
                        style={{
                          backgroundColor: darkMode ? '#3949ab' : '#1e88e5',
                          color: '#fff',
                        }}
                      >
                        Upload Images
                      </Button>
                    </label>
                    <Button
                      variant="contained"
                      startIcon={<AssessmentIcon />}
                      onClick={handleSubmit}
                      style={{
                        background: darkMode
                          ? 'linear-gradient(45deg, #1e3c72 30%, #2a5298 90%)'
                          : 'linear-gradient(45deg, #64b5f6 30%, #42a5f5 90%)',
                        color: '#fff',
                      }}
                      disabled={loading || selectedFiles.length === 0}
                    >
                      {loading ? <CircularProgress size={24} color="inherit" /> : 'Classify'}
                    </Button>
                  </Box>
                  {loading && <LinearProgress />}
                </CardContent>
              </Card>
            </Container>
          }
        />
        <Route
          path="/dashboard"
          element={
            <Container>
              <Tabs value={currentTab} onChange={handleTabChange} centered>
                <Tab label="New Predictions" />
                <Tab label="Archived Predictions" />
              </Tabs>
              {currentTab === 0 && (
                <Box>
                  <Typography variant="h4">New Prediction Results</Typography>
                  <TextareaAutosize
                    minRows={10}
                    style={{ width: '100%', padding: '1rem' }}
                    value={results}
                    readOnly
                    placeholder="New predictions will appear here..."
                  />
                </Box>
              )}
              {currentTab === 1 && (
                <Box>
                  <Typography variant="h4">Archived Predictions</Typography>
                  {oldResults.map((result, index) => (
                    <Box key={index} marginY={2} padding={2} border="1px solid #ddd">
                      <Typography variant="body1">
                        <strong>{result.timestamp.toString()}</strong>
                      </Typography>
                      <TextareaAutosize
                        minRows={5}
                        style={{ width: '100%', padding: '0.5rem' }}
                        value={result.results}
                        readOnly
                      />
                    </Box>
                  ))}
                </Box>
              )}
            </Container>
          }
        />
      </Routes>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={4000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
      />
    </>
  );
}

export default App;
