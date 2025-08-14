import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper, 
  Grid, 
  CircularProgress,
  AppBar,
  Toolbar,
  ThemeProvider,
  createMuiTheme
} from '@material-ui/core';
import ImageUploader from './components/ImageUploader';
import ModelSelector from './components/ModelSelector';
import ResultDisplay from './components/ResultDisplay';
import LlmAnalysis from './components/LlmAnalysis';
import OpenAIChat from './components/OpenAIChat';
import './App.css';

// Create a theme
const theme = createMuiTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
});

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [modelsStatus, setModelsStatus] = useState({
    yolo: false,
    detr: false,
    vit: false
  });

  // Check API status on component mount
  useEffect(() => {
    fetch('/api/status')
      .then(response => response.json())
      .then(data => {
        setModelsStatus(data.models);
      })
      .catch(err => {
        console.error('Error checking API status:', err);
        setError('Error connecting to the backend API. Please make sure the server is running.');
      });
  }, []);

  const handleImageUpload = (image) => {
    setSelectedImage(image);
    setResults(null);
    setError(null);
  };

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setResults(null);
    setError(null);
  };

  const processImage = async () => {
    if (!selectedImage || !selectedModel) {
      setError('Please select both an image and a model');
      return;
    }

    setIsProcessing(true);
    setError(null);

    // Create form data for the image
    const formData = new FormData();
    formData.append('image', selectedImage);

    let endpoint = '';
    switch (selectedModel) {
      case 'yolo':
        endpoint = '/api/detect/yolo';
        break;
      case 'detr':
        endpoint = '/api/detect/detr';
        break;
      case 'vit':
        endpoint = '/api/classify/vit';
        break;
      default:
        setError('Invalid model selection');
        setIsProcessing(false);
        return;
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setResults({ model: selectedModel, data });
    } catch (err) {
      console.error('Error processing image:', err);
      setError(`Error processing image: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box style={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" style={{ flexGrow: 1 }}>
              Multi-Model Object Detection Demo
            </Typography>
          </Toolbar>
        </AppBar>
        <Container maxWidth="lg" style={{ marginTop: theme.spacing(4), marginBottom: theme.spacing(4) }}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper style={{ padding: theme.spacing(2) }}>
                <Typography variant="h5" gutterBottom>
                  Upload an image to see how each model performs!
                </Typography>
                <Typography variant="body1" paragraph>
                  This demo showcases three different object detection and image classification models:
                </Typography>
                <Typography variant="body1" component="div">
                  <ul>
                    <li><strong>YOLOv8</strong>: Fast and accurate object detection</li>
                    <li><strong>DETR</strong>: DEtection TRansformer for object detection</li>
                    <li><strong>ViT</strong>: Vision Transformer for image classification</li>
                  </ul>
                </Typography>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <ImageUploader onImageUpload={handleImageUpload} />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <ModelSelector 
                onModelSelect={handleModelSelect} 
                onProcess={processImage}
                isProcessing={isProcessing}
                modelsStatus={modelsStatus}
                selectedModel={selectedModel}
                imageSelected={!!selectedImage}
              />
            </Grid>
            
            {error && (
              <Grid item xs={12}>
                <Paper style={{ padding: theme.spacing(2), backgroundColor: '#ffebee' }}>
                  <Typography color="error">{error}</Typography>
                </Paper>
              </Grid>
            )}
            
            {isProcessing && (
              <Grid item xs={12} style={{ textAlign: 'center', margin: `${theme.spacing(4)}px 0` }}>
                <CircularProgress />
                <Typography variant="h6" style={{ marginTop: theme.spacing(2) }}>
                  Processing image...
                </Typography>
              </Grid>
            )}
            
            {results && (
              <>
                <Grid item xs={12}>
                  <ResultDisplay results={results} />
                </Grid>
                <Grid item xs={12}>
                  <LlmAnalysis visionResults={results.data} model={results.model} />
                </Grid>
              </>
            )}

            {/* OpenAI Chat section at the end */}
            <Grid item xs={12}>
              <OpenAIChat />
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
