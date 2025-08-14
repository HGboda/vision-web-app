import React, { useState } from 'react';
import { 
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Divider
} from '@material-ui/core';

function OpenAIChat() {
  const [model, setModel] = useState('gpt-4o-mini');
  const [apiKey, setApiKey] = useState('');
  const [system, setSystem] = useState('You are a helpful assistant.');
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onSend = async () => {
    setError('');
    setResponse('');
    const p = (prompt || '').trim();
    if (!p) { setError('Please enter a question.'); return; }

    setLoading(true);
    try {
      const res = await fetch('/api/openai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          prompt: p,
          model: (model || '').trim() || 'gpt-4o-mini',
          api_key: (apiKey || undefined),
          system: (system || undefined)
        })
      });

      if (!res.ok) {
        let txt = await res.text();
        try { txt = JSON.stringify(JSON.parse(txt), null, 2); } catch {}
        throw new Error(txt);
      }
      const data = await res.json();
      const meta = `Model: ${data.model} | Latency: ${data.latency_sec}s` + (data.usage ? ` | Usage: ${JSON.stringify(data.usage)}` : '');
      setResponse((data.response || '(Empty response)') + '\n\n---\n' + meta);
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
        OpenAI Chat (OpenAI API)
      </Typography>
      <Typography variant="body2" color="textSecondary" gutterBottom>
        If the server env var OPENAI_API_KEY is set, the API Key field is optional.
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <TextField
            label="Model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            fullWidth
            variant="outlined"
            size="small"
          />
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
        <Grid item xs={12}>
          <TextField
            label="System Prompt (optional)"
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            fullWidth
            variant="outlined"
            size="small"
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
