/**
 * app.js
 * Frontend client logic for the Random Forest AI vs Human Text Detector.
 */

let cachedSamples = {};

// Switch navigation tabs
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

  if (tabId === 'detector') {
    document.getElementById('tabBtnDetector').classList.add('active');
    document.getElementById('paneDetector').classList.add('active');
  } else if (tabId === 'benchmarks') {
    document.getElementById('tabBtnBenchmarks').classList.add('active');
    document.getElementById('paneBenchmarks').classList.add('active');
    loadMetrics();
  } else if (tabId === 'theory') {
    document.getElementById('tabBtnTheory').classList.add('active');
    document.getElementById('paneTheory').classList.add('active');
  }
}

// Update word and character counts
function updateWordCount() {
  const text = document.getElementById('inputText').value.trim();
  const charCount = text.length;
  const words = text ? text.split(/\s+/).filter(Boolean) : [];
  const wordCount = words.length;

  document.getElementById('charCount').textContent = `${charCount} characters`;
  document.getElementById('wordCount').textContent = `${wordCount} words`;
}

// Clear text input
function clearInput() {
  document.getElementById('inputText').value = '';
  updateWordCount();
  document.getElementById('placeholderState').style.display = 'block';
  document.getElementById('outputContent').style.display = 'none';
}

// Fetch preset samples from server
async function fetchPresetSamples() {
  try {
    const res = await fetch('/api/samples');
    if (res.ok) {
      cachedSamples = await res.json();
    }
  } catch (err) {
    console.error('Failed to load sample texts:', err);
  }
}

// Load a specific preset sample
function loadPreset(key) {
  if (cachedSamples[key]) {
    document.getElementById('inputText').value = cachedSamples[key];
    updateWordCount();
    analyzeText();
  }
}

// Analyze text using Random Forest via backend API
async function analyzeText() {
  const text = document.getElementById('inputText').value.trim();

  if (!text) {
    alert('Please enter or paste some text first, or click one of the preset sample buttons!');
    return;
  }

  const analyzeBtn = document.getElementById('analyzeBtn');
  const btnIcon = document.getElementById('btnIcon');
  const btnText = document.getElementById('btnText');

  // Loading state
  analyzeBtn.disabled = true;
  btnIcon.innerHTML = '<span class="spinner"></span>';
  btnText.textContent = 'Random Forest Classifying...';

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to analyze text.');
    }

    renderPrediction(data);

  } catch (err) {
    alert('Error during analysis: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    btnIcon.textContent = '⚡';
    btnText.textContent = 'Analyze Text';
  }
}

// Render prediction output
function renderPrediction(data) {
  document.getElementById('placeholderState').style.display = 'none';
  const outputContent = document.getElementById('outputContent');
  outputContent.style.display = 'block';

  const isHuman = data.prediction_code === 0;
  const verdictCard = document.getElementById('verdictCard');
  const verdictTitle = document.getElementById('verdictTitle');
  const verdictBadge = document.getElementById('verdictBadge');
  const verdictConfidence = document.getElementById('verdictConfidence');

  verdictCard.className = `verdict-card ${isHuman ? 'human' : 'ai'}`;
  verdictTitle.textContent = data.prediction;
  verdictBadge.textContent = isHuman ? 'Natural Human Writing' : 'AI-Generated Text';
  verdictConfidence.textContent = `Random Forest Confidence: ${data.confidence}%`;

  // Probability bars
  document.getElementById('probHumanLabel').textContent = `Human: ${data.prob_human}%`;
  document.getElementById('probAiLabel').textContent = `AI: ${data.prob_ai}%`;
  document.getElementById('probFillHuman').style.width = `${data.prob_human}%`;
  document.getElementById('probFillAi').style.width = `${data.prob_ai}%`;

  // Handcrafted Features
  const f = data.features;
  document.getElementById('featLexDiv').textContent = f.lexical_diversity.toFixed(2);
  document.getElementById('featBurstiness').textContent = `${f.sentence_length_std.toFixed(1)} words`;
  document.getElementById('featWordLen').textContent = `${f.avg_word_length.toFixed(1)} chars`;
  document.getElementById('featFkGrade').textContent = `Grade ${f.flesch_kincaid_grade.toFixed(1)}`;

  // Explainability List
  const listEl = document.getElementById('explanationsList');
  listEl.innerHTML = '';
  if (data.explanations && data.explanations.length > 0) {
    data.explanations.forEach(exp => {
      const li = document.createElement('li');
      li.className = 'explanation-item';
      li.innerHTML = `<span>🔍</span> <span>${exp}</span>`;
      listEl.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.className = 'explanation-item';
    li.innerHTML = `<span>ℹ️</span> <span>Balanced statistical metrics observed.</span>`;
    listEl.appendChild(li);
  }

  // Smooth scroll to results on mobile
  if (window.innerWidth < 960) {
    outputContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// Load Random Forest evaluation metrics
let metricsLoaded = false;
async function loadMetrics() {
  if (metricsLoaded) return;
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) throw new Error('Could not fetch metrics');
    const data = await res.json();

    if (data.accuracy) {
      document.getElementById('metricAccuracy').textContent = `${(data.accuracy * 100).toFixed(2)}%`;
      document.getElementById('metricPrecision').textContent = data.precision.toFixed(4);
      document.getElementById('metricRecall').textContent = data.recall.toFixed(4);
      document.getElementById('metricF1').textContent = data.f1_score.toFixed(4);
      document.getElementById('metricCV').textContent = `${data.cv_f1_mean.toFixed(4)}`;
    }
    metricsLoaded = true;
  } catch (err) {
    console.error('Error loading metrics:', err);
  }
}

// On DOM load
document.addEventListener('DOMContentLoaded', () => {
  fetchPresetSamples();
  updateWordCount();
});
