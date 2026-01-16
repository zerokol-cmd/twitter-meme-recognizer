// background.js

// Configuration
const SERVER_URL = 'http://localhost:5000';
const TELEGRAM_API_ENDPOINT = '/api/telegram/analyze';

// Initialize storage on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    analysisHistory: [],
    serverUrl: SERVER_URL
  });
  console.log('[Background] Service worker installed and initialized');
});

// Message Listener
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Matches the structure sent from content.js
  if (message.action === 'SAVE_IMAGE') {
    
    const { id, base64, timestamp } = message.payload;

    console.log(`[Background] Processing Message ID: ${id}`);

    // Perform async analysis
    analyzeImage(id, base64, timestamp)
      .then(result => {
        sendResponse({ success: true, result: result });
      })
      .catch(error => {
        console.error('[Background] Analysis error:', error);
        sendResponse({ success: false, error: error.message });
      });

    // Return true to indicate we will sendResponse asynchronously
    return true; 
  }
});

/**
 * Sends image data to local server and saves history
 */
async function analyzeImage(messageId, base64Data, timestamp) {
  try {
    const payload = {
      messageId: messageId,
      image: base64Data,
      timestamp: timestamp || new Date().toISOString()
    };

    // 1. Send to Local API
    const response = await fetch(`${SERVER_URL}${TELEGRAM_API_ENDPOINT}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const apiResult = await response.json();

    // 2. Save to Extension History (Limit to last 50 items to save memory)
    await updateHistory({
      messageId,
      timestamp,
      status: 'success',
      apiResult
    });

    // 3. Notify Popup (if open)
    chrome.runtime.sendMessage({
      type: 'ANALYSIS_RESULT',
      data: apiResult,
      messageId: messageId
    }).catch(() => {
      // Ignore error if popup is closed
    });

    return apiResult;

  } catch (error) {
    // Log failure to history
    await updateHistory({
      messageId,
      timestamp,
      status: 'failed',
      error: error.message
    });
    throw error;
  }
}

/**
 * Helper to update local storage history
 */
async function updateHistory(record) {
  try {
    const data = await chrome.storage.local.get('analysisHistory');
    let history = data.analysisHistory || [];
    
    // Add new record to beginning
    history.unshift(record);
    
    // Keep only last 50
    if (history.length > 50) {
      history = history.slice(0, 50);
    }
    
    await chrome.storage.local.set({ analysisHistory: history });
  } catch (e) {
    console.warn('[Background] Failed to save history:', e);
  }
}