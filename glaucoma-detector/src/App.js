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
  Dialog,
  DialogActions,
  DialogContent,
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
  const [imageViewerOpen, setImageViewerOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

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
    setLoading(true);
    setProgress(0);

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('images', file);
    });

    try {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', backendURL);

      // Update progress
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentCompleted = Math.round((event.loaded / event.total) * 100);
          setProgress(percentCompleted);
        }
      };

      xhr.onload = async () => {
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText);
          const formattedResults = data.results
            .map(
              (result) =>
                `File: ${result.file}\n  vCDR: ${result.vCDR}\n  Prediction: ${result.prediction}\n`
            )
            .join('\n');
          setResults(formattedResults);
          setUploadSuccess(true);
          setSnackbarMessage('Classification completed successfully!');
          setOldResults((prev) => [
            ...prev,
            { timestamp: new Date(), results: formattedResults },
          ]);
        } else {
          const errorData = JSON.parse(xhr.responseText);
          setSnackbarMessage('Error: ' + errorData.error);
        }
        setSnackbarOpen(true);
        setLoading(false);
      };

      xhr.onerror = () => {
        setSnackbarMessage('Upload failed. Please try again.');
        setSnackbarOpen(true);
        setLoading(false);
      };

      xhr.send(formData);
    } catch (error) {
      setSnackbarMessage('Error: ' + error.message);
      setSnackbarOpen(true);
      setLoading(false);
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

  const handleImageClick = (image) => {
    setSelectedImage(image);
    setImageViewerOpen(true);
  };

  const handleCloseImageViewer = () => {
    setImageViewerOpen(false);
    setSelectedImage(null);
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
            <Container maxWidth="md" style={{ marginTop: '2rem' }}>
              <Card
                style={{
                  borderRadius: '15px',
                  padding: '2rem',
                  backgroundColor: darkMode ? '#283593' : '#e3f2fd',
                }}
              >
                <CardContent>
                  <Typography
                    variant="h4"
                    gutterBottom
                    style={{ color: darkMode ? '#fff' : '#000' }} // Set color to white during dark mode
                  >
                    Upload and Classify Images
                  </Typography>
                  <Box
                    display="flex"
                    justifyContent="center"
                    gap="1rem"
                    marginBottom="1rem"
                    style={{ color: darkMode ? '#fff' : '#000' }} // Set text color to white during dark mode
                  >
                    <label htmlFor="file-upload" style={{ color: darkMode ? '#fff' : '#000' }}>
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
                  {selectedFiles.length > 0 && (
                    <Box marginTop="2rem">
                      <Typography
                        variant="h6"
                        style={{ color: darkMode ? '#fff' : '#000' }} // Set text color to white during dark mode
                      >
                        Uploaded Files:
                      </Typography>
                      <Box maxHeight="200px" overflow="auto">
                        <ul style={{ color: darkMode ? '#fff' : '#000' }}>
                          {selectedFiles.map((file, index) => (
                            <li
                              key={index}
                              style={{ cursor: 'pointer', color: darkMode ? '#fff' : '#000' }}
                              onClick={() => handleImageClick(file)}
                            >
                              {file.name}
                            </li>
                          ))}
                        </ul>
                      </Box>
                    </Box>
                  )}
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
                  <Box
                    style={{
                      maxHeight: '400px',  // Set max height for the scrollable area
                      overflowY: 'auto',   // Enable vertical scrolling when content overflows
                      padding: '1rem',
                      border: '1px solid #ddd', // Optional: adds a border around the box
                    }}
                  >
                    <TextareaAutosize
                      minRows={10}
                      style={{
                        width: '100%',
                        padding: '1rem',
                        border: 'none',
                        background: 'transparent',
                      }}
                      value={results}
                      readOnly
                      placeholder="New predictions will appear here..."
                    />
                  </Box>
                </Box>
              )}
              {currentTab === 1 && (
                <Box>
                  <Typography variant="h4" style={{ color: '#000', marginBottom: '1rem' }}>
                    Archived Predictions
                  </Typography>
                  {oldResults.map((result, index) => (
                    <Box
                      key={index}
                      marginY={2}
                      padding={2}
                      border="1px solid #ddd"
                      borderRadius="8px" // Make the box slightly rounded
                      style={{
                        backgroundColor: '#f5f5f5', // Light background color
                        marginBottom: '1rem',
                      }}
                    >
                      <Typography variant="body1" style={{ color: '#000', fontWeight: 'bold' }}>
                        {result.timestamp.toString()}
                      </Typography>
                      <Box
                        maxHeight="300px" // Adjust maxHeight for scrollable area
                        overflow="auto"   // Enable scroll when content overflows
                        style={{
                          marginTop: '1rem',
                          backgroundColor: '#fff', // White background for the scrollable area
                          padding: '1rem',
                          borderRadius: '8px', // Rounded corners for consistency
                          border: '1px solid #ccc', // Light border
                        }}
                      >
                        <TextareaAutosize
                          minRows={5}
                          style={{
                            width: '100%',
                            padding: '0.5rem',
                            backgroundColor: '#fff', // White background for Textarea
                            color: '#000', // Black text color
                            border: '1px solid #ccc', // Light border
                            borderRadius: '8px', // Round corners for Textarea
                          }}
                          value={result.results}
                          readOnly
                        />
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}


            </Container>
          }
        />
      </Routes>

      {/* Image Viewer Modal */}
      <Dialog open={imageViewerOpen} onClose={handleCloseImageViewer}>
        <DialogContent>
          <img
            src={selectedImage ? URL.createObjectURL(selectedImage) : ''}
            alt="Selected"
            style={{ maxWidth: '100%', maxHeight: '400px', objectFit: 'contain' }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseImageViewer}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
      />
    </>
  );
}

export default App;
