# Telegram Message Analyzer Chrome Extension

A Chrome extension that automatically analyzes images in Telegram Web messages and classifies them using a deep learning model.

## Features

- 📸 **Auto-detection**: Automatically detects images in Telegram messages
- 🔄 **Real-time analysis**: Sends images to the classification server for analysis
- 📊 **Results display**: Shows classification results with confidence scores
- 💾 **History tracking**: Keeps a history of analyzed images and their results
- ⚙️ **Configurable**: Allows setting custom server URL

## Installation

### Prerequisites

1. **Chrome/Chromium browser**
2. **Python 3.8+** for the server
3. **Flask** and **torch** installed on your system

### Step 1: Install Python Dependencies

```bash
pip install flask flask-cors torch torchvision pillow matplotlib numpy
```

### Step 2: Load the Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **Load unpacked**
4. Navigate to the `plugin/telegram` folder and select it
5. The extension should now appear in your extensions list

### Step 3: Start the Server

```bash
# Run the Flask API server
python test.py

# Or run with Gradio UI
python test.py --gradio
```

The server will start on `http://localhost:5000`

## Usage

1. **Ensure the server is running** on your local machine
2. **Go to** `https://web.telegram.org/a/` in Chrome
3. **Open a conversation** with messages containing images
4. The extension will **automatically detect and analyze** images
5. **Click the extension icon** to see the analysis results in the popup

## API Endpoints

### 1. Health Check
```http
GET /api/status
```

**Response:**
```json
{
  "status": "ok",
  "service": "Twitter Image Classifier",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

### 2. Single Image Analysis
```http
POST /api/analyze
Content-Type: application/json

{
  "image": "data:image/png;base64,..."
}
```

Or with multipart form data:
```http
POST /api/analyze
Content-Type: multipart/form-data

image: <binary_image_data>
```

**Response:**
```json
{
  "success": true,
  "class": "twitter",
  "confidence": 0.95,
  "probabilities": {
    "twitter": 0.95,
    "else": 0.05
  },
  "timestamp": "2024-01-15T10:30:00.000000",
  "model": "resnet152"
}
```

### 3. Telegram Extension Analysis
```http
POST /api/telegram/analyze
Content-Type: application/json

{
  "image": "data:image/png;base64,...",
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

**Response:** Same as single image analysis

### 4. Batch Image Analysis
```http
POST /api/analyze-batch
Content-Type: application/json

{
  "images": [
    {
      "image": "data:image/png;base64,...",
      "id": "img1"
    },
    {
      "image": "data:image/png;base64,...",
      "id": "img2"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "results": [
    {
      "id": "img1",
      "class": "twitter",
      "confidence": 0.95,
      "probabilities": {...}
    },
    {
      "id": "img2",
      "class": "else",
      "confidence": 0.82,
      "probabilities": {...}
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

## Configuration

In the extension popup:
1. **Server URL**: Set the custom server URL (default: `http://localhost:5000`)
2. **Save Settings**: Save your configuration
3. **Clear History**: Remove all analysis history

## Project Structure

```
plugin/telegram/
├── manifest.json      # Extension manifest
├── content.js         # Content script for Telegram Web
├── background.js      # Background service worker
├── popup.html         # Popup UI
├── popup.js          # Popup logic
└── README.md         # This file
```

## How It Works

1. **Content Script** (`content.js`):
   - Monitors Telegram messages for new content
   - Detects images in messages
   - Extracts images and converts them to base64

2. **Background Script** (`background.js`):
   - Receives image data from content script
   - Sends images to the Flask server for analysis
   - Stores results in chrome.storage

3. **Popup** (`popup.html` + `popup.js`):
   - Displays analysis results and history
   - Shows server connection status
   - Allows configuration of server URL

4. **Flask Server** (`test.py`):
   - Receives image data via API
   - Loads pre-trained ResNet152 model
   - Performs image classification
   - Returns probabilities for all classes

## Troubleshooting

### Extension not detecting images
- Ensure you're on `https://web.telegram.org/a/`
- Check that the extension is enabled in Chrome settings
- Open the Developer Console (F12) and check for errors

### Server connection failed
- Ensure `python test.py` is running
- Check that the server URL is correctly set in the extension settings
- Default should be `http://localhost:5000`
- Make sure your firewall allows localhost connections

### Images not being analyzed
- Check the browser console for JavaScript errors
- Verify the Flask server is responding with `GET /api/status`
- Ensure the model weights file exists (`twitter_classifier.pth`)

## Notes

- Images are processed locally on your machine
- The extension does not store or transmit images anywhere except to your local server
- Make sure the model weights file (`twitter_classifier.pth`) is in the same directory as `test.py`
- The extension works best with clear, well-lit images

## License

This project is for personal use. Modify as needed for your requirements.
