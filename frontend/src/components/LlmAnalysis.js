import React, { useState } from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  TextField, 
  Button, 
  CircularProgress,
  Divider
} from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  paper: {
    padding: theme.spacing(2),
    marginTop: theme.spacing(2)
  },
  marginBottom: {
    marginBottom: theme.spacing(2)
  },
  dividerMargin: {
    margin: `${theme.spacing(2)}px 0`
  },
  responseBox: {
    padding: theme.spacing(2),
    backgroundColor: '#f5f5f5',
    borderRadius: theme.shape.borderRadius,
    marginTop: theme.spacing(2),
    whiteSpace: 'pre-wrap'
  },
  buttonProgress: {
    marginLeft: theme.spacing(1)
  }
}));

const LlmAnalysis = ({ visionResults, model }) => {
  const classes = useStyles();
  const [userQuery, setUserQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);

  // Format time for display
  const formatTime = (ms) => {
    if (ms === undefined || ms === null || isNaN(ms)) return '-';
    const num = Number(ms);
    if (num < 1000) return `${num.toFixed(2)} ms`;
    return `${(num / 1000).toFixed(2)} s`;
  };

  const handleAnalyze = async () => {
    if (!userQuery.trim()) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          visionResults: visionResults,
          userQuery: userQuery
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setAnalysisResult(data);
      }
    } catch (err) {
      console.error('Error analyzing with LLM:', err);
      setError(`Error analyzing with LLM: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (!visionResults) return null;

  return (
    <Paper className={classes.paper}>
      <Typography variant="h6" gutterBottom>
        Ask AI about the {model === 'vit' ? 'Classification' : 'Detection'} Results
      </Typography>
      
      <Typography variant="body2" className={classes.marginBottom}>
        Ask a question about the detected objects or classifications to get an AI-powered analysis.
      </Typography>
      
      <TextField
        fullWidth
        label="Your question about the image"
        variant="outlined"
        value={userQuery}
        onChange={(e) => setUserQuery(e.target.value)}
        disabled={isAnalyzing}
        className={classes.marginBottom}
        placeholder={model === 'vit' 
          ? "E.g., What category does this image belong to?" 
          : "E.g., How many people are in this image?"}
      />
      
      <Button 
        variant="contained" 
        color="primary"
        onClick={handleAnalyze}
        disabled={isAnalyzing || !userQuery.trim()}
      >
        Analyze with AI
        {isAnalyzing && <CircularProgress size={24} className={classes.buttonProgress} />}
      </Button>
      
      {error && (
        <Box mt={2}>
          <Typography color="error">{error}</Typography>
        </Box>
      )}
      
      {analysisResult && (
        <>
          <Divider className={classes.dividerMargin} />
          
          <Typography variant="subtitle1" gutterBottom>
            AI Analysis:
          </Typography>
          
          <Box className={classes.responseBox}>
            <Typography variant="body1">
              {analysisResult.response}
            </Typography>
          </Box>
          
          {analysisResult.performance && (
            <Box mt={1}>
              <Typography variant="body2" color="textSecondary">
                Analysis time: {formatTime(analysisResult.performance.inference_time)} on {analysisResult.performance.device}
              </Typography>
            </Box>
          )}
        </>
      )}
    </Paper>
  );
};

export default LlmAnalysis;
