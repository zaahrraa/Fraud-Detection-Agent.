const uploadForm = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const resultsSection = document.getElementById('results');
const resultSummary = document.getElementById('resultSummary');
const resultMetrics = document.getElementById('resultMetrics');

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  statusEl.textContent = 'Uploading dataset and training models...';
  statusEl.classList.remove('error');
  resultsSection.classList.add('hidden');

  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) {
    statusEl.textContent = 'Please choose a CSV file first.';
    statusEl.classList.add('error');
    return;
  }

  const formData = new FormData();
  formData.append('dataset', fileInput.files[0]);

  try {
    const response = await fetch('/api/train', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok || !data.success) {
      statusEl.textContent = data.error || 'Failed to train models.';
      statusEl.classList.add('error');
      return;
    }

    renderResults(data);
    statusEl.textContent = 'Training complete! Scroll down to see results.';
    statusEl.classList.remove('error');
  } catch (error) {
    statusEl.textContent = 'Unexpected error: ' + error.message;
    statusEl.classList.add('error');
  }
});

function renderResults(data) {
  resultsSection.classList.remove('hidden');
  resultSummary.innerHTML = '';
  resultMetrics.innerHTML = '';

  const summaryTiles = [
    { title: 'Total Rows', value: data.summary.rows },
    { title: 'Total Columns', value: data.summary.columns },
    { title: 'Legitimate', value: data.summary.legit_count },
    { title: 'Fraudulent', value: data.summary.fraud_count },
    { title: 'Fraud %', value: `${data.summary.fraud_percentage.toFixed(3)}%` },
    { title: 'Balanced Rows', value: data.balanced_rows },
  ];

  summaryTiles.forEach((tile) => {
    const div = document.createElement('div');
    div.className = 'result-tile';
    div.innerHTML = `<h3>${tile.title}</h3><p>${tile.value}</p>`;
    resultSummary.appendChild(div);
  });

  Object.values(data.metrics).forEach((report) => {
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <h3>${report.name}</h3>
      <ul class="metric-list">
        <li><strong>Accuracy:</strong> ${report.accuracy.toFixed(4)}</li>
        <li><strong>Precision:</strong> ${report.precision.toFixed(4)}</li>
        <li><strong>Recall:</strong> ${report.recall.toFixed(4)}</li>
        <li><strong>F1 Score:</strong> ${report.f1_score.toFixed(4)}</li>
        <li><strong>ROC AUC:</strong> ${report.roc_auc.toFixed(4)}</li>
      </ul>
    `;
    resultMetrics.appendChild(card);
  });
}
