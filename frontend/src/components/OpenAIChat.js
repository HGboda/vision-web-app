import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Divider,
  MenuItem,
  Select,
  FormControl,
  InputLabel
} from '@material-ui/core';

// Props:
// - imageBase64: data URL string of the currently uploaded image (preferred for searchType=image)
function OpenAIChat({ imageBase64 }) {
  const [apiKey, setApiKey] = useState('');
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Vision RAG parameters
  const [searchType, setSearchType] = useState('image');
  const [objectId, setObjectId] = useState('');
  const [className, setClassName] = useState('');
  const [nResults, setNResults] = useState(5);

  const onSend = async () => {
    setError('');
    setResponse('');
    const q = (prompt || '').trim();
    if (!q) { setError('질문을 입력하세요.'); return; }

    // Build request body for /api/vision-rag/query
    const body = {
      userQuery: q,
      searchType: searchType,
      n_results: Number(nResults) || 5,
    };
    if (apiKey) body.api_key = apiKey;
    if (searchType === 'image') {
      if (!imageBase64) { setError('이미지가 필요합니다. 먼저 이미지를 업로드하세요.'); return; }
      body.image = imageBase64;
    } else if (searchType === 'object') {
      if (!objectId.trim()) { setError('objectId를 입력하세요.'); return; }
      body.objectId = objectId.trim();
    } else if (searchType === 'class') {
      if (!className.trim()) { setError('class_name을 입력하세요.'); return; }
      body.class_name = className.trim();
    }

    setLoading(true);
    try {
      const res = await fetch('/api/vision-rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        let txt = await res.text();
        try { txt = JSON.stringify(JSON.parse(txt), null, 2); } catch {}
        throw new Error(txt);
      }
      const data = await res.json();
      const meta = `Model: ${data.model || '-'} | Latency: ${data.latency_sec || '-'}s`;
      setResponse((data.answer || '(빈 응답)') + '\n\n---\n' + meta);
    } catch (e) {
      setError('Error: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const onClear = () => {
    setPrompt('');
    setResponse('');
    setError('');
  };

  return (
    <Paper style={{ padding: 16 }}>
      <Typography variant="h5" gutterBottom>
        Vision RAG (LangChain)
      </Typography>
      <Typography variant="body2" color="textSecondary" gutterBottom>
        서버에 OPENAI_API_KEY가 설정되어 있다면 API Key는 생략 가능합니다. 검색 유형을 선택하고 질문을 보내면, 벡터 DB에서 검색된 컨텍스트로 답변합니다.
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel id="search-type-label">Search Type</InputLabel>
            <Select
              labelId="search-type-label"
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              label="Search Type"
            >
              <MenuItem value="image">image (current upload)</MenuItem>
              <MenuItem value="object">object (objectId)</MenuItem>
              <MenuItem value="class">class (class_name)</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={6}>
          <TextField
            label="OpenAI API Key (optional)"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            fullWidth
            variant="outlined"
            size="small"
            type="password"
            placeholder="sk-..."
          />
        </Grid>

        {searchType === 'object' && (
          <Grid item xs={12} md={6}>
            <TextField
              label="objectId"
              value={objectId}
              onChange={(e) => setObjectId(e.target.value)}
              fullWidth
              variant="outlined"
              size="small"
            />
          </Grid>
        )}
        {searchType === 'class' && (
          <Grid item xs={12} md={6}>
            <TextField
              label="class_name"
              value={className}
              onChange={(e) => setClassName(e.target.value)}
              fullWidth
              variant="outlined"
              size="small"
              placeholder="e.g. person, car, dog"
            />
          </Grid>
        )}
        <Grid item xs={12} md={6}>
          <TextField
            label="n_results"
            value={nResults}
            onChange={(e) => setNResults(e.target.value)}
            fullWidth
            variant="outlined"
            size="small"
            type="number"
            inputProps={{ min: 1, max: 50 }}
          />
        </Grid>

        <Grid item xs={12}>
          <TextField
            label="User Question"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            fullWidth
            multiline
            rows={4}
            variant="outlined"
            placeholder={searchType === 'image' ? '업로드한 이미지 기반으로 답변해줘' : '검색된 객체 컨텍스트를 사용해 답변해줘'}
          />
        </Grid>
        {error && (
          <Grid item xs={12}>
            <Typography color="error">{error}</Typography>
          </Grid>
        )}
        <Grid item xs={12}>
          <div style={{ display: 'flex', gap: 8 }}>
            <Button color="primary" variant="contained" onClick={onSend} disabled={loading}>
              {loading ? 'Sending...' : 'Send Question'}
            </Button>
            <Button variant="outlined" onClick={onClear}>Clear</Button>
          </div>
        </Grid>
        <Grid item xs={12}>
          <Divider style={{ margin: '12px 0' }} />
          <Typography variant="subtitle2" color="textSecondary">Response</Typography>
          <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'ui-monospace, monospace' }}>{response}</pre>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default OpenAIChat;
