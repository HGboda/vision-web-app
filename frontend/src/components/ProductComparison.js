import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Grid, 
  Typography, 
  Button,
  CircularProgress,
  Tabs,
  Tab,
  Divider,
  Card,
  CardContent,
  CardMedia,
  IconButton,
  TextField
} from '@material-ui/core';
import { AddCircle, Delete, Compare, Search, Info } from '@material-ui/icons';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles((theme) => ({
  root: {
    marginTop: theme.spacing(3),
    marginBottom: theme.spacing(3),
  },
  imageContainer: {
    position: 'relative',
    minHeight: '360px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    border: '2px dashed #ccc',
    borderRadius: '8px',
    margin: theme.spacing(1),
    padding: theme.spacing(1),
    backgroundColor: '#fafafa',
    overflow: 'hidden',
  },
  progressLog: {
    marginTop: theme.spacing(2),
    height: '200px',
    overflowY: 'auto',
    backgroundColor: '#f8f9fa',
    padding: theme.spacing(1),
    borderRadius: '4px',
    fontFamily: 'monospace',
    fontSize: '0.9rem',
  },
  logEntry: {
    margin: '4px 0',
    padding: '2px 5px',
    borderLeft: '3px solid #ccc',
  },
  logEntryAgent: {
    borderLeft: '3px solid #2196f3',
  },
  logEntrySystem: {
    borderLeft: '3px solid #4caf50',
  },
  logEntryError: {
    borderLeft: '3px solid #f44336',
  },
  logTime: {
    color: '#666',
    fontSize: '0.8rem',
    marginRight: theme.spacing(1),
  },
  imagePreview: {
    width: '100%',
    height: 'auto',
    maxHeight: '60vh',
    objectFit: 'contain',
  },
  uploadIcon: {
    fontSize: '3rem',
    color: '#aaa',
  },
  uploadInput: {
    display: 'none',
  },
  deleteButton: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    '&:hover': {
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
    },
  },
  tabPanel: {
    padding: theme.spacing(2),
  },
  resultCard: {
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(2),
  },
  comparisonTable: {
    width: '100%',
    borderCollapse: 'collapse',
    '& th, & td': {
      border: '1px solid #ddd',
      padding: '8px',
      textAlign: 'left',
    },
    '& th': {
      backgroundColor: '#f2f2f2',
    },
    '& tr:nth-child(even)': {
      backgroundColor: '#f9f9f9',
    },
  },
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: theme.spacing(4),
  },
  highlight: {
    backgroundColor: '#e3f2fd',
    padding: theme.spacing(1),
    borderRadius: '4px',
    fontWeight: 'bold',
  }
}));

// 분석 유형 탭 패널 컴포넌트
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box p={3}>
          {children}
        </Box>
      )}
    </div>
  );
}

const ProductComparison = () => {
  const classes = useStyles();
  const [images, setImages] = useState([null, null]); // 최대 2개 이미지 저장
  const [imagePreviews, setImagePreviews] = useState([null, null]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [progressLogs, setProgressLogs] = useState([]);
  const logEndRef = useRef(null);

  // 탭 변경 핸들러
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // 이미지 업로드 핸들러
  const handleImageUpload = (event, index) => {
    const file = event.target.files[0];
    
    if (file) {
      // 이미지 미리보기 생성
      const reader = new FileReader();
      reader.onload = (e) => {
        const newPreviews = [...imagePreviews];
        newPreviews[index] = e.target.result;
        setImagePreviews(newPreviews);
      };
      reader.readAsDataURL(file);
      
      // 이미지 파일 상태 업데이트
      const newImages = [...images];
      newImages[index] = file;
      setImages(newImages);
      
      // 결과 및 오류 초기화
      setAnalysisResults(null);
      setError(null);
    }
  };

  // 이미지 삭제 핸들러
  const handleImageDelete = (index) => {
    const newImages = [...images];
    const newPreviews = [...imagePreviews];
    
    newImages[index] = null;
    newPreviews[index] = null;
    
    setImages(newImages);
    setImagePreviews(newPreviews);
    setAnalysisResults(null);
  };

  // 로그 추가 함수
  const addLog = (message, type = 'info') => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    const newLog = {
      time: timeStr,
      message,
      type // 'info', 'agent', 'system', 'error'
    };
    
    setProgressLogs(logs => [...logs, newLog]);
  };

  // 로그창 자동 스크롤
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [progressLogs]);

  // SSE 연결 함수
  const connectToSSE = (sessionId) => {
    const eventSource = new EventSource(`/api/product/compare/stream/${sessionId}`);
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('React SSE received:', data);
        
        if (data.message) {
          addLog(data.message, data.agent || 'info');
        } else if (data.status) {
          addLog(`Status: ${data.status}`, 'system');
        } else if (data.final_result) {
          console.log('Final result received:', data.final_result);
          setAnalysisResults(data.final_result);
          setIsProcessing(false);
          eventSource.close();
        } else if (data.error) {
          addLog(`Error: ${data.error}`, 'error');
          setIsProcessing(false);
          eventSource.close();
        }
      } catch (err) {
        console.error('SSE parsing error:', err);
        addLog(`Event processing error: ${err.message}`, 'error');
      }
    };
    
    eventSource.onerror = (err) => {
      addLog('Server connection lost. Please try again in a moment.', 'error');
      eventSource.close();
      setIsProcessing(false);
    };
    
    return eventSource;
  };

  // 제품 분석 처리 핸들러 (analysisType 강제 가능)
  const handleAnalysis = async (analysisOverride = null) => {
    // 유효성 검사: 최소 1개 이상의 이미지가 있어야 함
    if (!images[0] && !images[1]) {
      setError('Please upload at least one product image for analysis');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProgressLogs([]); // 로그 초기화
    addLog('Starting product analysis...', 'system');
    
    try {
      const formData = new FormData();
      
      // 업로드된 이미지만 FormData에 추가
      if (images[0]) {
        formData.append('image1', images[0]);
        addLog('Product A image uploaded.', 'info');
      }
      if (images[1]) {
        formData.append('image2', images[1]);
        addLog('Product B image uploaded.', 'info');
      }
      
      // 분석 타입 추가 (탭 인덱스로 구분) 혹은 명시적 override
      const analysisTypes = ['info', 'compare', 'value', 'recommend'];
      const analysisType = analysisOverride || analysisTypes[tabValue];
      formData.append('analysisType', analysisType);
      addLog(`Analysis type: ${analysisType === 'info' ? 'Product Information' : analysisType === 'compare' ? 'Performance Comparison' : analysisType === 'value' ? 'Value Analysis' : 'Purchase Recommendation'}`, 'system');
      
      // 백엔드 API 호출 (세션 시작)
      addLog('Initializing analysis session...', 'system');
      
      // Debug FormData contents
      for (let [key, value] of formData.entries()) {
        console.log('FormData:', key, value);
      }
      
      const response = await fetch('/api/product/compare/start', {
        method: 'POST',
        body: formData,
        credentials: 'include', // 세션 쿠키 포함
      });
      
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`HTTP error! Status: ${response.status} - ${errorText}`);
      }
      
      const data = await response.json();
      const sessionId = data.session_id;
      
      if (!sessionId) {
        throw new Error('Failed to receive session ID');
      }
      
      addLog(`Analysis session started. (Session ID: ${sessionId.substring(0, 8)}...)`, 'system');
      addLog('Agents are collaborating to analyze products. Please wait a moment...', 'system');
      
      // SSE 스트림 연결
      const eventSource = connectToSSE(sessionId);
      
      // 컴포넌트 언마운트시 연결 종료
      return () => {
        eventSource.close();
      };
      
    } catch (err) {
      console.error('제품 분석 오류:', err);
      addLog(`Error occurred: ${err.message}`, 'error');
      setError(`Error during analysis: ${err.message}`);
      setIsProcessing(false);
    }
  };

  // 분석 결과 렌더링 함수
  const renderAnalysisResults = () => {
    if (!analysisResults) return null;
    
    // 분석 유형에 따라 다른 결과 표시
    switch (tabValue) {
      case 0: // 제품 정보 탐색
        return (
          <Card className={classes.resultCard}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Product Information</Typography>
              <Divider style={{ margin: '8px 0 16px' }} />
              
              {analysisResults.productInfo && (
                <div>
                  {Object.entries(analysisResults.productInfo).map(([imageKey, productData], index) => (
                    <div key={imageKey} style={{ marginBottom: '24px' }}>
                      <Typography variant="subtitle1" gutterBottom>
                        <strong>Product {index + 1} ({imageKey})</strong>
                      </Typography>
                      
                      <Typography variant="body1">
                        <strong>Type:</strong> {productData.product_type || 'Unknown'}
                      </Typography>
                      
                      {productData.key_features && productData.key_features.length > 0 && (
                        <div style={{ marginTop: '12px' }}>
                          <Typography variant="subtitle2">Key Features:</Typography>
                          <ul>
                            {productData.key_features.map((feature, idx) => (
                              <li key={idx}>
                                <Typography variant="body2">{feature}</Typography>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {analysisResults.specifications && analysisResults.specifications[imageKey] && (
                        <div style={{ marginTop: '12px' }}>
                          <Typography variant="subtitle2">Specifications:</Typography>
                          <ul>
                            {Object.entries(analysisResults.specifications[imageKey].specifications || {}).map(([key, value]) => (
                              <li key={key}>
                                <Typography variant="body2">
                                  <strong>{key.replace('_', ' ')}:</strong> {Array.isArray(value) ? value.join(', ') : value}
                                </Typography>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        );
        
      case 1: // 제품 성능 비교
        return (
          <Card className={classes.resultCard}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Product Comparison Analysis</Typography>
              <Divider style={{ margin: '8px 0 16px' }} />
              
              {analysisResults.comparison && (
                <div>
                  <Typography variant="body1" style={{ whiteSpace: 'pre-line', marginBottom: '16px' }}>
                    {typeof analysisResults.comparison === 'string' 
                      ? analysisResults.comparison 
                      : JSON.stringify(analysisResults.comparison, null, 2)
                    }
                  </Typography>
                </div>
              )}
            </CardContent>
          </Card>
        );
        
      case 2: // 가격 대비 가치 분석
        return (
          <Card className={classes.resultCard}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Price-to-Value Analysis</Typography>
              <Divider style={{ margin: '8px 0 16px' }} />
              
              {analysisResults.valueAnalysis && (
                <div>
                  <Typography variant="body1" style={{ whiteSpace: 'pre-line', marginBottom: '16px' }}>
                    {typeof analysisResults.valueAnalysis === 'string' 
                      ? analysisResults.valueAnalysis 
                      : JSON.stringify(analysisResults.valueAnalysis, null, 2)
                    }
                  </Typography>
                </div>
              )}
            </CardContent>
          </Card>
        );
        
      case 3: // 최적 구매 추천
        return (
          <Card className={classes.resultCard}>
            <CardContent>
              <Typography variant="h6" gutterBottom>Purchase Recommendations</Typography>
              <Divider style={{ margin: '8px 0 16px' }} />
              
              {analysisResults.recommendation && (
                <div>
                  <Typography variant="body1" style={{ whiteSpace: 'pre-line', marginBottom: '16px' }}>
                    {typeof analysisResults.recommendation === 'string' 
                      ? analysisResults.recommendation 
                      : JSON.stringify(analysisResults.recommendation, null, 2)
                    }
                  </Typography>
                </div>
              )}
              
              {/* Fallback: Display all analysis results as JSON if no specific data found */}
              {!analysisResults.recommendation && !analysisResults.valueAnalysis && !analysisResults.comparison && !analysisResults.productInfo && (
                <div>
                  <Typography variant="body1" style={{ whiteSpace: 'pre-line', marginBottom: '16px' }}>
                    {JSON.stringify(analysisResults, null, 2)}
                  </Typography>
                </div>
              )}
            </CardContent>
          </Card>
        );
        
      default:
        return null;
    }
  };

  return (
    <Paper className={classes.root}>
      <Box p={3}>
        <Typography variant="h5" gutterBottom>
          Product Comparison Analysis
        </Typography>
        <Typography variant="body1" paragraph>
          Upload product images to receive detailed information and comparison analysis.
          You can analyze various products including cars, smartphones, laptops, and more.
        </Typography>
        
        <Grid container spacing={3}>
          {/* 이미지 업로드 영역 */}
          {[0, 1].map((index) => (
            <Grid item xs={12} md={6} key={index}>
              <Box className={classes.imageContainer}>
                {imagePreviews[index] ? (
                  <>
                    <img 
                      src={imagePreviews[index]} 
                      alt={`Product ${index + 1}`}
                      className={classes.imagePreview}
                    />
                    <IconButton
                      className={classes.deleteButton}
                      onClick={() => handleImageDelete(index)}
                    >
                      <Delete />
                    </IconButton>
                  </>
                ) : (
                  <>
                    <input
                      accept="image/*"
                      className={classes.uploadInput}
                      id={`upload-image-${index}`}
                      type="file"
                      onChange={(e) => handleImageUpload(e, index)}
                    />
                    <label htmlFor={`upload-image-${index}`}>
                      <Box display="flex" flexDirection="column" alignItems="center">
                        <AddCircle className={classes.uploadIcon} />
                        <Typography variant="body2" style={{ marginTop: '8px' }}>
                          Upload {index === 0 ? 'Product A' : 'Product B'} Image
                        </Typography>
                      </Box>
                    </label>
                  </>
                )}
              </Box>
            </Grid>
          ))}

          {/* 다중 파일 업로드 (선택 사항): 두 장을 한 번에 업로드 */}
          <Grid item xs={12}>
            <input
              accept="image/*"
              className={classes.uploadInput}
              id="upload-both-images"
              type="file"
              multiple
              onChange={(e) => {
                const files = Array.from(e.target.files || []);
                if (!files.length) return;
                const newImages = [...images];
                const newPreviews = [...imagePreviews];
                files.slice(0, 2).forEach((file, idx) => {
                  const slot = idx; // 0,1 순서로 채움
                  newImages[slot] = file;
                  const reader = new FileReader();
                  reader.onload = (ev) => {
                    newPreviews[slot] = ev.target.result;
                    setImagePreviews([...newPreviews]);
                  };
                  reader.readAsDataURL(file);
                });
                setImages(newImages);
                setAnalysisResults(null);
                setError(null);
              }}
            />
            <label htmlFor="upload-both-images">
              <Button variant="text" color="default" component="span">
                Or select two files at once
              </Button>
            </label>
          </Grid>
          
          {/* 분석 유형 탭 */}
          <Grid item xs={12}>
            <Paper>
              <Tabs
                value={tabValue}
                onChange={handleTabChange}
                indicatorColor="primary"
                textColor="primary"
                centered
              >
                <Tab icon={<Info />} label="Product Info" />
                <Tab icon={<Compare />} label="Performance" disabled={!images[0] || !images[1]} />
                <Tab icon={<Search />} label="Value Analysis" />
                <Tab label="Recommendations" />
              </Tabs>
              
              <Box p={2} display="flex" justifyContent="center" gridGap={12}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => handleAnalysis(null)}
                  disabled={isProcessing || (!images[0] && !images[1])}
                  startIcon={isProcessing ? <CircularProgress size={24} /> : null}
                >
                  {isProcessing ? 'Analyzing...' : 'Start Analysis'}
                </Button>
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={() => handleAnalysis('compare')}
                  disabled={isProcessing || !images[0] || !images[1]}
                  startIcon={<Compare />}
                >
                  Compare Products
                </Button>
              </Box>
              
              {/* 오류 메시지 표시 */}
              {error && (
                <Box p={2} bgcolor="#ffebee" borderRadius="4px" mb={2}>
                  <Typography color="error">{error}</Typography>
                </Box>
              )}
            </Paper>
          </Grid>
          
          {/* 진행 과정 로그 표시 */}
          <Grid item xs={12}>
            <Paper>
              <Box p={2}>
                <Typography variant="h6" gutterBottom>
                  Analysis Progress
                </Typography>
                <Box className={classes.progressLog}>
                  {progressLogs.length === 0 ? (
                    <Typography variant="body2" color="textSecondary" style={{padding: '10px'}}>
                      Progress details will appear here when analysis starts.
                    </Typography>
                  ) : (
                    progressLogs.map((log, index) => (
                      <Box 
                        key={index} 
                        className={`${classes.logEntry} ${log.type === 'agent' ? classes.logEntryAgent : log.type === 'system' ? classes.logEntrySystem : log.type === 'error' ? classes.logEntryError : ''}`}
                      >
                        <span className={classes.logTime}>[{log.time}]</span>
                        {log.message}
                      </Box>
                    ))
                  )}
                  <div ref={logEndRef} />
                </Box>
              </Box>
            </Paper>
          </Grid>

          {/* 결과 표시 영역 */}
          <Grid item xs={12}>
            {isProcessing ? (
              <Box className={classes.loadingContainer}>
                <CircularProgress />
                <Typography variant="h6" style={{ marginTop: '16px' }}>
                  Analyzing Products...
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Product recognition, information retrieval, and comparison analysis in progress.
                </Typography>
              </Box>
            ) : renderAnalysisResults()}
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default ProductComparison;
