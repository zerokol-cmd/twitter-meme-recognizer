// Popup script
// Manages the popup UI and displays analysis results

let analysisHistory = [];

// Load settings and history when popup opens
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  loadHistory();
  checkServerStatus();
  setupEventListeners();
  
  // Check server status every 5 seconds
  setInterval(checkServerStatus, 5000);
});

function setupEventListeners() {
  document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
  document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);
}

function loadSettings() {
  chrome.storage.local.get(['serverUrl'], (result) => {
    if (result.serverUrl) {
      document.getElementById('serverUrl').value = result.serverUrl;
    }
  });
}

function saveSettings() {
  const serverUrl = document.getElementById('serverUrl').value;
  
  if (!serverUrl) {
    alert('Please enter a server URL');
    return;
  }
  
  chrome.storage.local.set({ serverUrl }, () => {
    alert('Settings saved!');
  });
}

function loadHistory() {
  chrome.storage.local.get(['analysisHistory'], (result) => {
    analysisHistory = result.analysisHistory || [];
    displayResults();
  });
}

function clearHistory() {
  if (confirm('Clear all analysis history?')) {
    chrome.storage.local.set({ analysisHistory: [] }, () => {
      analysisHistory = [];
      displayResults();
      alert('History cleared!');
    });
  }
}

function displayResults() {
  const resultsList = document.getElementById('resultsList');
  
  if (analysisHistory.length === 0) {
    resultsList.innerHTML = '<div class="empty-state">No analysis results yet<br/>Images will appear here as they are analyzed</div>';
    return;
  }
  
  resultsList.innerHTML = analysisHistory
    .slice()
    .reverse()
    .slice(0, 20) // Show last 20 results
    .map((item, index) => {
      const probabilities = item.result?.probabilities || {};
      const maxClass = Object.entries(probabilities).sort(([,a], [,b]) => b - a)[0];
      
      return `
        <div class="result-item">
          <span class="result-class">${maxClass ? maxClass[0] : 'Unknown'}</span>
          <div class="result-prob">
            ${maxClass ? `Confidence: ${(maxClass[1] * 100).toFixed(1)}%` : 'Processing...'}
            <br/>
            <small>${new Date(item.timestamp).toLocaleTimeString()}</small>
          </div>
        </div>
      `;
    })
    .join('');
}

async function checkServerStatus() {
  const serverUrl = document.getElementById('serverUrl').value || 'http://localhost:5000';
  const statusIndicator = document.getElementById('serverStatus');
  const statusText = document.getElementById('serverStatusText');
  const extensionStatus = document.getElementById('extensionStatus');
  
  try {
    const response = await fetch(`${serverUrl}/api/status`, { timeout: 3000 });
    if (response.ok) {
      statusIndicator.classList.remove('disconnected');
      statusIndicator.classList.add('connected');
      statusText.textContent = 'Server connected';
    } else {
      throw new Error('Server returned error');
    }
  } catch (error) {
    statusIndicator.classList.remove('connected');
    statusIndicator.classList.add('disconnected');
    statusText.textContent = 'Server disconnected';
  }
  
  // Extension status is always active
  extensionStatus.classList.add('connected');
}

// Listen for analysis results from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'ANALYSIS_RESULT') {
    analysisHistory.unshift({
      timestamp: request.timestamp,
      result: request.data
    });
    
    // Keep only last 100 results
    if (analysisHistory.length > 100) {
      analysisHistory = analysisHistory.slice(0, 100);
    }
    
    chrome.storage.local.set({ analysisHistory });
    displayResults();
  }
});
