// content.js

// --- CONFIGURATION ---
const TARGET_CHAT_ID = "5219816270"; 

// --- STATE MANAGEMENT ---
const processedMessages = new Set();
const PROCESSING_QUEUE = new Set();
let scanInterval = null;

// --- QUEUE SYSTEM ---
const taskQueue = []; 
let isBusy = false;

// --- 1. CSS INJECTION (Ghost Mode) ---
function injectStyles() {
  const styleId = 'tg-bot-style';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      body.tg-bot-reacting .bubble.menu-container {
        opacity: 0 !important;
        transform: translate(-100vw, -100vh) !important;
        pointer-events: none !important;
      }
    `;
    document.head.appendChild(style);
  }
}

// --- 2. QUEUE PROCESSOR ---
async function processQueue() {
  if (isBusy || taskQueue.length === 0) return;

  isBusy = true;
  const task = taskQueue.shift();

  try {
    if (task.action === 'delete') {
      await performUiDelete(task.id);
    } else if (task.action === 'react') {
      await performUiReaction(task.id, task.type);
    }
  } catch (err) {
    console.warn(`[TG-Queue] Failed task for ${task.id}:`, err);
  } finally {
    setTimeout(() => {
      isBusy = false;
      processQueue();
    }, 1000); 
  }
}

// --- 3. HELPER: FIND TARGET ELEMENT ---
// Finds the clickable DOM element for a given Message ID (handles Single & Album)
function findTargetElement(msgId) {
  // 1. Try finding it as an Album Item (Specific Image)
  const albumItem = document.getElementById(`album-media-message-${msgId}`);
  if (albumItem) return albumItem;

  // 2. Try finding it as a Standard Message
  const stdMessage = document.querySelector(`.Message[data-message-id="${msgId}"]`);
  if (stdMessage) {
    // Return the content wrapper or the message itself
    return stdMessage.querySelector('.message-content') || stdMessage;
  }

  return null;
}

// --- 4. UI ACTION: DELETE ---
function performUiDelete(msgId) {
  return new Promise((resolve) => {
    if (!window.location.hash.includes(TARGET_CHAT_ID)) return resolve();

    document.body.classList.add('tg-bot-reacting');

    const targetNode = findTargetElement(msgId);
    if (!targetNode) {
      document.body.classList.remove('tg-bot-reacting');
      return resolve(); 
    }

    const contextMenuEvent = new MouseEvent('contextmenu', {
      bubbles: true, cancelable: true, view: window, buttons: 2, clientX: 0, clientY: 0
    });
    targetNode.dispatchEvent(contextMenuEvent);

    setTimeout(() => {
      const deleteBtn = document.querySelector('.MenuItem.destructive');
      
      if (deleteBtn) {
        deleteBtn.click();
        setTimeout(() => {
          const confirmBtn = document.querySelector('.popup-button.danger') || 
                             document.querySelector('.Button.danger') || 
                             document.querySelector('button.confirm-dialog-button');

          if (confirmBtn) {
            confirmBtn.click();
            console.log(`[TG-UI] DELETED Message ID ${msgId}`);
          } else {
            document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13 }));
          }
          document.body.classList.remove('tg-bot-reacting');
          resolve();
        }, 400);
      } else {
        document.body.click();
        document.body.classList.remove('tg-bot-reacting');
        resolve();
      }
    }, 400);
  });
}

// --- 5. UI ACTION: REACT ---
function performUiReaction(msgId, reactionIndex) {
  return new Promise((resolve) => {
    if (!window.location.hash.includes(TARGET_CHAT_ID)) return resolve();

    document.body.classList.add('tg-bot-reacting');

    const targetNode = findTargetElement(msgId);
    if (!targetNode) {
      document.body.classList.remove('tg-bot-reacting');
      return resolve(); 
    }

    const contextMenuEvent = new MouseEvent('contextmenu', {
      bubbles: true, cancelable: true, view: window, buttons: 2, clientX: 0, clientY: 0
    });
    targetNode.dispatchEvent(contextMenuEvent);

    setTimeout(() => {
      const reactionContainer = document.querySelector('.ReactionSelector__reactions');

      if (reactionContainer && reactionContainer.children.length > reactionIndex) {
        const reactionBtn = reactionContainer.children[reactionIndex];
        if (reactionBtn) {
          reactionBtn.click();
          console.log(`[TG-UI] Reacted to ID ${msgId} (Index ${reactionIndex})`);
        }
      } else {
        document.body.click(); 
      }

      document.body.classList.remove('tg-bot-reacting');
      resolve();

    }, 400);
  });
}

// --- 6. IMAGE CONVERSION ---
function imageToBase64(img) {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL('image/jpeg', 0.85));
    } catch (e) {
      reject(e);
    }
  });
}

// --- 7. HELPER: PROCESS SINGLE ITEM ---
function processItem(id, imgElement) {
  if (processedMessages.has(id) || PROCESSING_QUEUE.has(id)) return;
  if (taskQueue.some(t => t.id === id)) return;

  if (!imgElement.complete || imgElement.naturalHeight === 0) return;

  PROCESSING_QUEUE.add(id);

  imageToBase64(imgElement)
    .then(base64Data => {
      if (!chrome.runtime?.id) throw new Error('Extension context invalidated');

      chrome.runtime.sendMessage({
        action: 'SAVE_IMAGE',
        payload: {
          id: id,
          timestamp: new Date().toISOString(),
          base64: base64Data
        }
      }, (response) => {
        PROCESSING_QUEUE.delete(id);
        if (chrome.runtime.lastError) return;

        processedMessages.add(id);

        if (response && response.result) {
          const resultClass = response.result.class;
          
          if (resultClass === 'twitter') {
            console.log(`[TG-Scan] ID ${id} is Twitter -> QUEUE DELETE.`);
            taskQueue.push({ id: id, action: 'delete' });
          } else {
            console.log(`[TG-Scan] ID ${id} is Other -> QUEUE POOP REACTION.`);
            taskQueue.push({ id: id, action: 'react', type: 1 });
          }
          processQueue();
        }
      });
    })
    .catch(err => {
      PROCESSING_QUEUE.delete(id);
      if (err.message?.includes('invalidated')) clearInterval(scanInterval);
    });
}

// --- 8. SCANNING LOGIC ---
function scanForImages() {
  if (!chrome.runtime?.id) {
    if (scanInterval) clearInterval(scanInterval);
    return;
  }

  if (!window.location.hash.includes(TARGET_CHAT_ID)) return;

  // Select all main message containers
  const messages = document.querySelectorAll('.Message[data-message-id]');

  messages.forEach(node => {
    // Check if this container (or album) already has reactions. 
    // If so, we assume all contents are "done" to avoid re-reacting.
    const hasReactions = !!node.querySelector('.Reactions');
    
    // --- ALBUM DETECTION ---
    const albumItems = node.querySelectorAll('[id^="album-media-message-"]');
    
    if (albumItems.length > 0) {
      // It is an album (Multiple Images)
      albumItems.forEach(item => {
        // Extract ID from "album-media-message-19541"
        const idParts = item.id.split('-');
        const subId = idParts[idParts.length - 1];

        if (hasReactions) {
          processedMessages.add(subId); // Skip logic
          return;
        }

        const img = item.querySelector('img.full-media');
        if (img) processItem(subId, img);
      });
    } else {
      // It is a Single Message
      const msgId = node.getAttribute('data-message-id');
      if (hasReactions) {
        processedMessages.add(msgId);
        return;
      }

      const img = node.querySelector('img.full-media');
      if (img) processItem(msgId, img);
    }
  });
}

// --- 9. INITIALIZATION ---
function init() {
  console.log(`[TG-Scan] Mode: Albums + Singles. Target: ${TARGET_CHAT_ID}`);
  injectStyles(); 
  if (scanInterval) clearInterval(scanInterval);
  scanForImages();
  scanInterval = setInterval(scanForImages, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}