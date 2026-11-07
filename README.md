# Vision Lab — Real-time Object Detection

A browser-based computer vision project for the **Artificial Intelligence & Machine Learning** category. It uses TensorFlow.js and the COCO-SSD model to identify everyday objects in a live camera feed.

## Features

- Live object detection with labeled bounding boxes
- Confidence threshold control and live detection results
- Toggles for bounding boxes and labels
- Per-frame inference timing and object count
- On-device processing: camera video is not uploaded or saved
- Responsive layout for desktop, tablet, and mobile

## Run locally

Camera access requires a secure context. Use `localhost` rather than opening the HTML file directly:

1. Make sure Python is installed.
2. From this folder, run `python -m http.server 8000`.
3. Open `http://localhost:8000` in a recent browser.
4. Wait for the model status to say **AI model ready**, then select **Start camera** and allow camera access.

The TensorFlow.js scripts and model weights are loaded from jsDelivr, so an internet connection is needed on first load. Camera video and predictions remain in the browser.

## Model

The app loads the COCO-SSD `lite_mobilenet_v2` variant. COCO-SSD can recognize 80 common categories, including people, vehicles, household objects, and animals. Predictions below the selected confidence threshold are hidden.

## Troubleshooting

- **Camera permission denied:** Allow camera access in the browser's site settings and try again.
- **Camera unavailable:** Connect a webcam and close other apps that may be using it.
- **Model unavailable:** Check your internet connection and refresh the page.
- **Camera access unavailable:** Use `localhost` or serve the project over HTTPS.
