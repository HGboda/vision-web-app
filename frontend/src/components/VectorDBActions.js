import React, { useState } from 'react';
import { 
  Button, 
  Box, 
  Typography, 
  CircularProgress, 
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Card,
  CardMedia,
  CardContent,
  Chip
} from '@material-ui/core';
import { Alert } from '@material-ui/lab';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(2),
    padding: theme.spacing(2),
    backgroundColor: '#f5f5f5',
    borderRadius: theme.shape.borderRadius,
  },
  button: {
    marginRight: theme.spacing(2),
  },
  searchDialog: {
    minWidth: '500px',
  },
  formControl: {
    marginBottom: theme.spacing(2),
    minWidth: '100%',
  },
  searchResults: {
    marginTop: theme.spacing(2),
  },
  resultCard: {
    marginBottom: theme.spacing(2),
  },
  resultImage: {
    height: 140,
    objectFit: 'contain',
  },
  chip: {
    margin: theme.spacing(0.5),
  },
  similarityChip: {
    backgroundColor: theme.palette.primary.main,
    color: 'white',
  }
}));

const VectorDBActions = ({ results }) => {
  const classes = useStyles();
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [openSearchDialog, setOpenSearchDialog] = useState(false);
  const [searchType, setSearchType] = useState('image');
  const [searchClass, setSearchClass] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);
  
  // Extract model and data from results
  const { model, data } = results;
  
  // Handle saving to vector DB
  const handleSaveToVectorDB = async () => {
    setIsSaving(true);
    setSaveError(null);
    
    try {
      let response;
      
      if (model === 'vit') {
        // For ViT, save the whole image with classifications
        response = await fetch('/api/add-to-collection', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image: data.image,
            metadata: {
              model: 'vit',
              classifications: data.classifications
            }
          })
        });
      } else {
        // For YOLO and DETR, save detected objects
        response = await fetch('/api/add-detected-objects', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image: data.image,
            objects: data.detections,
            imageId: generateUUID()
          })
        });
      }
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 5000);
    } catch (err) {
      console.error('Error saving to vector DB:', err);
      setSaveError(`Error saving to vector DB: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };
  
  // Handle opening search dialog
  const handleOpenSearchDialog = () => {
    setOpenSearchDialog(true);
    setSearchResults([]);
    setSearchError(null);
  };
  
  // Handle closing search dialog
  const handleCloseSearchDialog = () => {
    setOpenSearchDialog(false);
  };
  
  // Handle search type change
  const handleSearchTypeChange = (event) => {
    setSearchType(event.target.value);
    setSearchResults([]);
    setSearchError(null);
  };
  
  // Handle search class change
  const handleSearchClassChange = (event) => {
    setSearchClass(event.target.value);
  };
  
  // Handle search
  const handleSearch = async () => {
    setIsSearching(true);
    setSearchError(null);
    
    try {
      let requestBody = {};
      
      if (searchType === 'image') {
        // Search by current image
        requestBody = {
          searchType: 'image',
          image: data.image,
          n_results: 5
        };
      } else {
        // Search by class name
        if (!searchClass.trim()) {
          throw new Error('Please enter a class name');
        }
        
        requestBody = {
          searchType: 'class',
          class_name: searchClass.trim(),
          n_results: 5
        };
      }
      
      const response = await fetch('/api/search-similar-objects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      console.log('Search API response:', result);
      
      // The backend responds with {success, searchType, results} structure, so extract only the results array
      if (result.success && Array.isArray(result.results)) {
        console.log('Setting search results array:', result.results);
        console.log('Results array length:', result.results.length);
        console.log('First result item:', result.results[0]);
        setSearchResults(result.results);
      } else {
        console.error('Unexpected API response format:', result);
        throw new Error('Unexpected API response format');
      }
    } catch (err) {
      console.error('Error searching vector DB:', err);
      setSearchError(`Error searching vector DB: ${err.message}`);
    } finally {
      setIsSearching(false);
    }
  };
  
  // Generate UUID for image ID
  const generateUUID = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  };
  
  // Render search results
  const renderSearchResults = () => {
    console.log('Rendering search results:', searchResults);
    console.log('Search results length:', searchResults.length);
    
    if (searchResults.length === 0) {
      console.log('No results to render');
      return (
        <Typography variant="body1">No results found.</Typography>
      );
    }
    
    return (
      <Grid container spacing={2}>
        {searchResults.map((result, index) => {
          const similarity = (1 - result.distance) * 100;
          
          return (
            <Grid item xs={12} sm={6} key={index}>
              <Card className={classes.resultCard}>
                {result.metadata && result.metadata.image_data ? (
                  <CardMedia
                    className={classes.resultImage}
                    component="img"
                    height="200"
                    image={`data:image/jpeg;base64,${result.metadata.image_data}`}
                    alt={result.metadata && result.metadata.class ? result.metadata.class : 'Object'}
                  />
                ) : (
                  <Box 
                    className={classes.resultImage}
                    style={{ 
                      backgroundColor: '#f0f0f0', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      height: 200
                    }}
                  >
                    <Typography variant="body2" color="textSecondary">
                      {result.metadata && result.metadata.class ? result.metadata.class : 'Object'} Image
                    </Typography>
                  </Box>
                )}
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="subtitle1">Result #{index + 1}</Typography>
                    <Chip 
                      label={`Similarity: ${similarity.toFixed(2)}%`}
                      className={classes.similarityChip}
                      size="small"
                    />
                  </Box>
                  <Typography variant="body2" color="textSecondary">
                    <strong>Class:</strong> {result.metadata.class || 'N/A'}
                  </Typography>
                  {result.metadata.confidence && (
                    <Typography variant="body2" color="textSecondary">
                      <strong>Confidence:</strong> {(result.metadata.confidence * 100).toFixed(2)}%
                    </Typography>
                  )}
                  <Typography variant="body2" color="textSecondary">
                    <strong>Object ID:</strong> {result.id}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    );
  };
  
  return (
    <Box className={classes.root}>
      <Typography variant="h6" gutterBottom>
        Vector Database Actions
      </Typography>
      
      <Box display="flex" alignItems="center" mb={2}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSaveToVectorDB}
          disabled={isSaving}
          className={classes.button}
        >
          {isSaving ? (
            <>
              <CircularProgress size={20} color="inherit" style={{ marginRight: 8 }} />
              Saving...
            </>
          ) : (
            'Save to Vector DB'
          )}
        </Button>
        
        <Button
          variant="outlined"
          color="primary"
          onClick={handleOpenSearchDialog}
          className={classes.button}
        >
          Search Similar
        </Button>
      </Box>
      
      {saveError && (
        <Alert severity="error" style={{ marginTop: 8 }}>
          {saveError}
        </Alert>
      )}
      
      <Snackbar open={saveSuccess} autoHideDuration={5000} onClose={() => setSaveSuccess(false)}>
        <Alert severity="success">
          {model === 'vit' ? (
            'Image and classifications successfully saved to vector DB!'
          ) : (
            'Detected objects successfully saved to vector DB!'
          )}
        </Alert>
      </Snackbar>
      
      {/* Search Dialog */}
      <Dialog
        open={openSearchDialog}
        onClose={handleCloseSearchDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Search Vector Database</DialogTitle>
        <DialogContent>
          <FormControl className={classes.formControl}>
            <InputLabel id="search-type-label">Search Type</InputLabel>
            <Select
              labelId="search-type-label"
              id="search-type"
              value={searchType}
              onChange={handleSearchTypeChange}
            >
              <MenuItem value="image">Search by Current Image</MenuItem>
              <MenuItem value="class">Search by Class Name</MenuItem>
            </Select>
          </FormControl>
          
          {searchType === 'class' && (
            <FormControl className={classes.formControl}>
              <TextField
                label="Class Name"
                value={searchClass}
                onChange={handleSearchClassChange}
                placeholder="e.g. person, car, dog..."
                fullWidth
              />
            </FormControl>
          )}
          
          {searchError && (
            <Alert severity="error" style={{ marginBottom: 16 }}>
              {searchError}
            </Alert>
          )}
          
          <Box className={classes.searchResults}>
            {isSearching ? (
              <Box display="flex" justifyContent="center" alignItems="center" p={4}>
                <CircularProgress />
                <Typography variant="body1" style={{ marginLeft: 16 }}>
                  Searching...
                </Typography>
              </Box>
            ) : (
              <>
                {console.log('Search dialog render - searchResults:', searchResults)}
                {searchResults.length > 0 ? renderSearchResults() : 
                  <Typography variant="body1">No results found. Please try another search.</Typography>
                }
              </>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseSearchDialog} color="default">
            Close
          </Button>
          <Button 
            onClick={handleSearch} 
            color="primary" 
            variant="contained"
            disabled={isSearching || (searchType === 'class' && !searchClass.trim())}
          >
            Search
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VectorDBActions;
