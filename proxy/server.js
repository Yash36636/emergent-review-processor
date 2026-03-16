/**
 * AI API Proxy — routes Groq/Grok calls through this server.
 * Keeps API keys server-side; avoids CORS and browser blocks.
 *
 * Run: node server.js
 * Then set USE_AI_PROXY=true in .env and restart the Flask app.
 */

require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ limit: '10mb', extended: true }));

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

app.post('/api/grok', async (req, res) => {
  if (!GROQ_API_KEY) {
    return res.status(400).json({
      choices: [{ message: { content: 'GROQ_API_KEY not set in .env' } }],
    });
  }

  try {
    const response = await fetch(GROQ_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json(
        data.error ? { error: data.error } : data
      );
    }

    res.json(data);
  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(500).json({
      choices: [{ message: { content: `Proxy error: ${err.message}` } }],
    });
  }
});

const PORT = process.env.PROXY_PORT || 3001;
app.listen(PORT, () => {
  console.log(`AI proxy running on http://localhost:${PORT}`);
  console.log(`  POST /api/grok → ${GROQ_URL}`);
  console.log(`  GROQ_API_KEY: ${GROQ_API_KEY ? 'set' : 'NOT SET'}`);
});
