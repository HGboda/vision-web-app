import React from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  List, 
  ListItem, 
  ListItemText, 
  Divider,
  Grid,
  Chip
} from '@material-ui/core';
import VectorDBActions from './VectorDBActions';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  paper: {
    padding: theme.spacing(2)
  },
  marginBottom: {
    marginBottom: theme.spacing(2)
  },
  resultImage: {
    maxWidth: '100%',
    maxHeight: '400px',
    objectFit: 'contain'
  },
  dividerMargin: {
    margin: `${theme.spacing(2)}px 0`
  },
  chipContainer: {
    display: 'flex',
    gap: theme.spacing(1),
    flexWrap: 'wrap'
  }
}));

const ResultDisplay = ({ results }) => {
  const classes = useStyles();
  if (!results) return null;
  
  const { model, data } = results;
  
  // Helper to format times nicely
  const formatTime = (ms) => {
    if (ms === undefined || ms === null || isNaN(ms)) return '-';
    const num = Number(ms);
    if (num < 1000) return `${num.toFixed(2)} ms`;
    return `${(num / 1000).toFixed(2)} s`;
  };
  
  // Check if there's an error
  if (data.error) {
    return (
      <Paper sx={{ p: 2, bgcolor: '#ffebee' }}>
        <Typography color="error">{data.error}</Typography>
      </Paper>
    );
  }

  // Display performance info
  const renderPerformanceInfo = () => {
    if (!data.performance) return null;
    
    return (
      <Box className="performance-info">
        <Divider className={classes.dividerMargin} />
        <Typography variant="body2">
          Inference time: {formatTime(data.performance.inference_time)} on {data.performance.device}
        </Typography>
      </Box>
    );
  };

  // Render for YOLO and DETR (object detection)
  if (model === 'yolo' || model === 'detr') {
    return (
      <Paper className={classes.paper}>
        <Typography variant="h6" gutterBottom>
          {model === 'yolo' ? 'YOLOv8' : 'DETR'} Detection Results
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            {data.image && (
              <Box className={classes.marginBottom}>
                <Typography variant="subtitle1" gutterBottom>
                  Detection Result
                </Typography>
                <img 
                  src={`data:image/png;base64,${data.image}`} 
                  alt="Detection Result" 
                  className={classes.resultImage}
                />
              </Box>
            )}
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box className={classes.marginBottom}>
              <Typography variant="subtitle1" gutterBottom>
                Detected Objects:
              </Typography>
              
              {data.detections && data.detections.length > 0 ? (
                <List>
                  {data.detections.map((detection, index) => (
                    <React.Fragment key={index}>
                      <ListItem>
                        <ListItemText 
                          primary={
                            <Box style={{ display: 'flex', alignItems: 'center' }}>
                              <Typography variant="body1" component="span">
                                {detection.class}
                              </Typography>
                              <Chip 
                                label={`${(detection.confidence * 100).toFixed(0)}%`}
                                size="small"
                                color="primary"
                                style={{ marginLeft: 8 }}
                              />
                            </Box>
                          } 
                          secondary={`Bounding Box: [${detection.bbox.join(', ')}]`} 
                        />
                      </ListItem>
                      {index < data.detections.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Typography variant="body1">No objects detected</Typography>
              )}
            </Box>
          </Grid>
        </Grid>
        
        {renderPerformanceInfo()}
        
        {/* Vector DB Actions for Object Detection */}
        <VectorDBActions results={results} />
      </Paper>
    );
  }
  
  // Render for ViT (classification)
  if (model === 'vit') {
    return (
      <Paper className={classes.paper}>
        <Typography variant="h6" gutterBottom>
          ViT Classification Results
        </Typography>
        
        <Typography variant="subtitle1" gutterBottom>
          Top Predictions:
        </Typography>
        
        {data.top_predictions && data.top_predictions.length > 0 ? (
          <List>
            {data.top_predictions.map((prediction, index) => (
              <React.Fragment key={index}>
                <ListItem>
                  <ListItemText 
                    primary={
                      <Box style={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="body1" component="span">
                          {prediction.rank}. {prediction.class}
                        </Typography>
                        <Chip 
                          label={`${(prediction.probability * 100).toFixed(1)}%`}
                          size="small"
                          color={index === 0 ? "primary" : "default"}
                          style={{ marginLeft: 8 }}
                        />
                      </Box>
                    } 
                  />
                </ListItem>
                {index < data.top_predictions.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        ) : (
          <Typography variant="body1">No classifications available</Typography>
        )}
        
        {renderPerformanceInfo()}
        
        {/* Vector DB Actions for ViT Classification */}
        <VectorDBActions results={results} />
      </Paper>
    );
  }
  
  return null;
};

export default ResultDisplay;
