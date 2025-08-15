import React, { useState, useRef } from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  Button, 
  IconButton 
} from '@material-ui/core';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
import DeleteIcon from '@material-ui/icons/Delete';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  paper: {
    padding: theme.spacing(2),
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    height: '100%',
    minHeight: 300,
    transition: 'all 0.3s ease'
  },
  dragActive: {
    border: '2px dashed #3f51b5',
    backgroundColor: 'rgba(63, 81, 181, 0.05)'
  },
  dragInactive: {
    border: '2px dashed #ccc',
    backgroundColor: 'white'
  },
  uploadBox: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    width: '100%',
    cursor: 'pointer'
  },
  uploadIcon: {
    fontSize: 60,
    color: '#3f51b5',
    marginBottom: theme.spacing(2)
  },
  supportText: {
    marginTop: theme.spacing(2)
  },
  previewBox: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    position: 'relative'
  },
  imageContainer: {
    position: 'relative',
    width: '100%',
    // Use viewport-based height so any aspect ratio fits inside
    height: '60vh',
    [theme.breakpoints.down('sm')]: {
      height: '45vh',
    },
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
    marginTop: theme.spacing(2),
  },
  deleteButton: {
    position: 'absolute',
    top: 0,
    right: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    '&:hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
    }
  }
}));

const ImageUploader = ({ onImageUpload }) => {
  const classes = useStyles();
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files[0]);
    }
  };

  const handleFiles = (file) => {
    if (file.type.startsWith('image/')) {
      setPreviewUrl(URL.createObjectURL(file));
      onImageUpload(file);
    } else {
      alert('Please upload an image file');
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleRemoveImage = () => {
    setPreviewUrl(null);
    onImageUpload(null);
    fileInputRef.current.value = "";
  };

  return (
    <Paper 
      className={`${classes.paper} ${dragActive ? classes.dragActive : classes.dragInactive}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleChange}
        style={{ display: 'none' }}
      />

      {!previewUrl ? (
        <Box 
          className={classes.uploadBox}
          onClick={onButtonClick}
        >
          <CloudUploadIcon className={classes.uploadIcon} />
          <Typography variant="h6" gutterBottom>
            Drag & Drop an image here
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            or
          </Typography>
          <Button
            variant="contained"
            color="primary"
            component="span"
            startIcon={<CloudUploadIcon />}
          >
            Browse Files
          </Button>
          <Typography variant="body2" color="textSecondary" className={classes.supportText}>
            Supported formats: JPG, PNG, GIF
          </Typography>
        </Box>
      ) : (
        <Box className={classes.previewBox}>
          <Typography variant="h6" gutterBottom>
            Preview
          </Typography>
          <Box className={classes.imageContainer}>
            <img
              src={previewUrl}
              alt="Preview"
              className="preview-image"
            />
            <IconButton
              aria-label="delete"
              className={classes.deleteButton}
              onClick={handleRemoveImage}
            >
              <DeleteIcon />
            </IconButton>
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default ImageUploader;
