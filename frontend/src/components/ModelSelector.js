import React from 'react';
import { 
  Grid, 
  Card, 
  CardContent, 
  CardActions, 
  Typography, 
  Button, 
  Chip,
  Box
} from '@material-ui/core';
import VisibilityIcon from '@material-ui/icons/Visibility';
import CategoryIcon from '@material-ui/icons/Category';
import PlayArrowIcon from '@material-ui/icons/PlayArrow';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  card: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  selectedCard: {
    border: '2px solid #3f51b5',
  },
  unavailableCard: {
    opacity: 0.6,
  },
  cardContent: {
    flexGrow: 1,
  },
  chipContainer: {
    marginBottom: theme.spacing(1.5),
  },
  successChip: {
    backgroundColor: '#34C759',
    color: '#fff',
  },
  errorChip: {
    backgroundColor: '#FF3B3F',
    color: '#fff',
  },
  modelType: {
    marginTop: theme.spacing(1),
  },
  processButton: {
    marginTop: theme.spacing(3),
    textAlign: 'center',
  }
}));

const ModelSelector = ({ 
  onModelSelect, 
  onProcess, 
  isProcessing, 
  modelsStatus, 
  selectedModel,
  imageSelected 
}) => {
  const classes = useStyles();
  
  const models = [
    {
      id: 'yolo',
      name: 'YOLOv8',
      description: 'Fast and accurate object detection',
      icon: <VisibilityIcon />,
      available: modelsStatus.yolo
    },
    {
      id: 'detr',
      name: 'DETR',
      description: 'DEtection TRansformer for object detection',
      icon: <VisibilityIcon />,
      available: modelsStatus.detr
    },
    {
      id: 'vit',
      name: 'ViT',
      description: 'Vision Transformer for image classification',
      icon: <CategoryIcon />,
      available: modelsStatus.vit
    }
  ];

  const handleModelClick = (modelId) => {
    if (models.find(m => m.id === modelId).available) {
      onModelSelect(modelId);
    }
  };

  return (
    <Box sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Select Model
      </Typography>
      
      <Grid container spacing={2}>
        {models.map((model) => (
          <Grid item xs={12} sm={4} key={model.id}>
            <Card 
              className={`
                ${classes.card} 
                ${selectedModel === model.id ? classes.selectedCard : ''} 
                ${!model.available ? classes.unavailableCard : ''}
              `}
              onClick={() => handleModelClick(model.id)}
            >
              <CardContent className={classes.cardContent}>
                <Box sx={{ mb: 2, color: 'primary' }}>
                  {model.icon}
                </Box>
                <Typography variant="h5" component="div" gutterBottom>
                  {model.name}
                </Typography>
                <div className={classes.chipContainer}>
                  {model.available ? (
                    <Chip 
                      label="Available" 
                      className={classes.successChip}
                      size="small" 
                    />
                  ) : (
                    <Chip 
                      label="Not Available" 
                      className={classes.errorChip}
                      size="small" 
                    />
                  )}
                </div>
                <Typography variant="body2" color="textSecondary">
                  {model.description}
                </Typography>
              </CardContent>
              <CardActions>
                <Button 
                  size="small" 
                  onClick={() => handleModelClick(model.id)}
                  disabled={!model.available}
                  color={selectedModel === model.id ? "primary" : "default"}
                  variant={selectedModel === model.id ? "contained" : "outlined"}
                  fullWidth
                >
                  {selectedModel === model.id ? 'Selected' : 'Select'}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <div className={classes.processButton}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<PlayArrowIcon />}
          onClick={onProcess}
          disabled={!selectedModel || !imageSelected || isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Process Image'}
        </Button>
      </div>
    </Box>
  );
};

export default ModelSelector;
