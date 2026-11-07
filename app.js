(() => {
  "use strict";

  const video = document.getElementById("camera");
  const canvas = document.getElementById("detectionCanvas");
  const context = canvas.getContext("2d");
  const startButton = document.getElementById("startButton");
  const stopButton = document.getElementById("stopButton");
  const modelState = document.getElementById("modelState");
  const modelStateText = document.getElementById("modelStateText");
  const cameraCaption = document.getElementById("cameraCaption");
  const cameraStage = document.getElementById("cameraStage");
  const cameraPlaceholder = document.getElementById("cameraPlaceholder");
  const cameraError = document.getElementById("cameraError");
  const cameraModeText = document.getElementById("cameraModeText");
  const resolutionText = document.getElementById("resolutionText");
  const fpsText = document.getElementById("fpsText");
  const livePill = document.getElementById("livePill");
  const confidenceSlider = document.getElementById("confidenceSlider");
  const confidenceValue = document.getElementById("confidenceValue");
  const boxesToggle = document.getElementById("boxesToggle");
  const labelsToggle = document.getElementById("labelsToggle");
  const detectionCount = document.getElementById("detectionCount");
  const detectionList = document.getElementById("detectionList");
  const emptyDetections = document.getElementById("emptyDetections");
  const objectsFound = document.getElementById("objectsFound");
  const inferenceTime = document.getElementById("inferenceTime");
  const analysisStatus = document.getElementById("analysisStatus");
  const footerDot = document.querySelector(".footer-dot");

  let model = null;
  let mediaStream = null;
  let animationFrame = null;
  let lastInferenceAt = 0;
  let inferenceInProgress = false;
  let cameraStarting = false;
  let latestPredictions = [];
  let lastRenderKey = "";
  let disposed = false;

  const BOX_COLORS = ["#9f88ff", "#63d9a1", "#66b5ff", "#ffbd69", "#ff7f9b", "#4ed8d1"];

  function setModelState(state, message) {
    modelState.classList.remove("is-ready", "is-error");
    const indicator = modelState.querySelector(".state-indicator");
    indicator.className = `state-indicator ${state}`;
    if (state === "ready") modelState.classList.add("is-ready");
    if (state === "error") modelState.classList.add("is-error");
    modelStateText.textContent = message;
  }

  function updateSliderFill() {
    const minimum = Number(confidenceSlider.min);
    const maximum = Number(confidenceSlider.max);
    const value = Number(confidenceSlider.value);
    const percentage = ((value - minimum) / (maximum - minimum)) * 100;
    confidenceSlider.style.background = `linear-gradient(90deg, #967bf1 ${percentage}%, #353642 ${percentage}%)`;
    confidenceValue.value = `${value}%`;
    confidenceValue.textContent = `${value}%`;
  }

  function showCameraError(message) {
    cameraError.textContent = message;
    cameraError.hidden = false;
  }

  function clearCameraError() {
    cameraError.textContent = "";
    cameraError.hidden = true;
  }

  function loadExternalScript(source, globalName) {
    if (window[globalName]) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = source;
      script.async = true;
      const timeout = window.setTimeout(() => {
        script.remove();
        reject(new Error("Loading the AI libraries timed out. Check your internet connection and refresh the page."));
      }, 30000);

      script.addEventListener("load", () => {
        window.clearTimeout(timeout);
        if (window[globalName]) {
          resolve();
        } else {
          reject(new Error("An AI library loaded without initializing. Refresh the page and try again."));
        }
      }, { once: true });
      script.addEventListener("error", () => {
        window.clearTimeout(timeout);
        script.remove();
        reject(new Error("Could not load the AI libraries. Check your internet connection and refresh the page."));
      }, { once: true });
      document.head.append(script);
    });
  }

  async function loadModel() {
    try {
      await loadExternalScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js", "tf");
      await window.tf.ready();
      await loadExternalScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3", "cocoSsd");
      model = await window.cocoSsd.load({ base: "lite_mobilenet_v2" });
      if (disposed) return;
      setModelState("ready", "AI model ready");
      startButton.disabled = false;
      analysisStatus.textContent = "Model ready · Waiting for camera";
    } catch (error) {
      if (disposed) return;
      setModelState("error", "AI model unavailable");
      analysisStatus.textContent = "Model failed to load";
      showCameraError(error instanceof Error ? error.message : "The AI model could not be loaded.");
      console.error("Could not load the object detection model:", error);
    }
  }

  async function startCamera() {
    if (cameraStarting || mediaStream) return;
    cameraStarting = true;
    startButton.disabled = true;
    clearCameraError();
    cameraCaption.textContent = "Requesting camera access…";

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera access is unavailable in this browser. Open this page on localhost or use HTTPS.");
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" } }
      });
      video.srcObject = mediaStream;
      await video.play();

      cameraPlaceholder.hidden = true;
      video.style.display = "block";
      cameraStage.classList.add("is-live");
      livePill.classList.add("active");
      cameraModeText.textContent = "CAMERA ON";
      cameraCaption.textContent = "Live feed · Processed on device";
      resolutionText.textContent = `${video.videoWidth} × ${video.videoHeight}`;
      fpsText.textContent = "Detecting objects";
      startButton.disabled = true;
      stopButton.disabled = false;
      analysisStatus.textContent = "Analyzing live camera feed";
      footerDot.classList.remove("inactive");
      lastInferenceAt = 0;
      animationFrame = requestAnimationFrame(detectionLoop);
    } catch (error) {
      stopCamera();
      cameraCaption.textContent = "Camera is not connected";
      const message = error && error.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in your browser settings, then try again."
        : error && error.name === "NotFoundError"
          ? "No camera was found. Connect a camera and try again."
          : error instanceof Error
            ? error.message
            : "Could not start the camera. Check the browser permission and try again.";
      showCameraError(message);
      console.error("Could not start the camera:", error);
    } finally {
      cameraStarting = false;
      startButton.disabled = !model || Boolean(mediaStream);
    }
  }

  function stopCamera() {
    if (animationFrame !== null) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    if (mediaStream) {
      for (const track of mediaStream.getTracks()) track.stop();
      mediaStream = null;
    }
    video.pause();
    video.srcObject = null;
    video.style.display = "none";
    context.clearRect(0, 0, canvas.width, canvas.height);
    cameraPlaceholder.hidden = false;
    cameraStage.classList.remove("is-live");
    livePill.classList.remove("active");
    cameraModeText.textContent = "CAMERA OFF";
    resolutionText.textContent = "— × —";
    fpsText.textContent = "Waiting for camera";
    cameraCaption.textContent = "Camera is not connected";
    startButton.disabled = !model;
    stopButton.disabled = true;
    inferenceInProgress = false;
    latestPredictions = [];
    lastRenderKey = "";
    renderDetections([]);
    inferenceTime.textContent = "—";
    analysisStatus.textContent = model ? "Model ready · Waiting for camera" : "Waiting for AI model";
    footerDot.classList.add("inactive");
  }

  function detectionLoop(timestamp) {
    if (!mediaStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      animationFrame = requestAnimationFrame(detectionLoop);
      return;
    }

    if (!inferenceInProgress && timestamp - lastInferenceAt >= 100) {
      lastInferenceAt = timestamp;
      inferenceInProgress = true;
      detectObjects();
    } else {
      drawPredictions(latestPredictions);
    }
    animationFrame = requestAnimationFrame(detectionLoop);
  }

  async function detectObjects() {
    const startedAt = performance.now();
    try {
      const predictions = await model.detect(video);
      if (!mediaStream) return;
      latestPredictions = predictions.filter(
        (prediction) => prediction.score >= Number(confidenceSlider.value) / 100
      );
      inferenceTime.textContent = String(Math.round(performance.now() - startedAt));
      objectsFound.textContent = String(latestPredictions.length);
      detectionCount.textContent = String(latestPredictions.length);
      fpsText.textContent = `${latestPredictions.length} object${latestPredictions.length === 1 ? "" : "s"} in view`;
      drawPredictions(latestPredictions);
      renderDetections(latestPredictions);
    } catch (error) {
      if (mediaStream) {
        showCameraError("Detection stopped unexpectedly. Stop the camera and try again.");
        console.error("Object detection failed:", error);
        stopCamera();
      }
    } finally {
      inferenceInProgress = false;
    }
  }

  function drawPredictions(predictions) {
    if (!video.videoWidth || !video.videoHeight) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!boxesToggle.checked && !labelsToggle.checked) return;

    predictions.forEach((prediction, index) => {
      const [x, y, width, height] = prediction.bbox;
      const color = BOX_COLORS[index % BOX_COLORS.length];
      if (boxesToggle.checked) {
        context.strokeStyle = color;
        context.lineWidth = Math.max(2, canvas.width / 550);
        context.strokeRect(x, y, width, height);
        const cornerLength = Math.min(width, height, canvas.width * 0.035);
        context.lineWidth = Math.max(3, canvas.width / 300);
        for (const [cornerX, cornerY, directionX, directionY] of [
          [x, y, 1, 1],
          [x + width, y, -1, 1],
          [x, y + height, 1, -1],
          [x + width, y + height, -1, -1]
        ]) {
          context.beginPath();
          context.moveTo(cornerX, cornerY + directionY * cornerLength);
          context.lineTo(cornerX, cornerY);
          context.lineTo(cornerX + directionX * cornerLength, cornerY);
          context.stroke();
        }
      }
      if (labelsToggle.checked) {
        const label = `${prediction.class}  ${Math.round(prediction.score * 100)}%`;
        const fontSize = Math.max(13, canvas.width / 65);
        context.font = `600 ${fontSize}px "DM Sans", sans-serif`;
        const labelWidth = context.measureText(label).width + 14;
        const labelHeight = fontSize + 10;
        const labelY = Math.max(0, y - labelHeight);
        context.fillStyle = color;
        context.beginPath();
        context.roundRect(x, labelY, labelWidth, labelHeight, 4);
        context.fill();
        context.fillStyle = "#10111b";
        context.textBaseline = "middle";
        context.fillText(label, x + 7, labelY + labelHeight / 2);
      }
    });
  }

  function renderDetections(predictions) {
    const renderKey = predictions
      .map((prediction) => `${prediction.class}:${Math.round(prediction.score * 100)}`)
      .sort()
      .join("|");
    if (renderKey === lastRenderKey) return;
    lastRenderKey = renderKey;

    for (const item of detectionList.querySelectorAll(".detection-item")) item.remove();
    emptyDetections.hidden = predictions.length > 0;
    detectionCount.textContent = String(predictions.length);
    objectsFound.textContent = String(predictions.length);

    const sortedPredictions = [...predictions].sort((a, b) => b.score - a.score);
    sortedPredictions.forEach((prediction, index) => {
      const percentage = Math.round(prediction.score * 100);
      const item = document.createElement("div");
      item.className = "detection-item";
      item.innerHTML = `
        <span class="object-symbol" aria-hidden="true"><svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="3" /><path d="M9 9h6v6H9z" /></svg></span>
        <span><span class="object-name"></span><span class="object-class">Object ${String(index + 1).padStart(2, "0")}</span></span>
        <span class="confidence-meter"><span class="confidence-bar"><span></span></span><span class="confidence-number">${percentage}%</span></span>`;
      item.querySelector(".object-name").textContent = prediction.class;
      item.querySelector(".object-name").style.color = BOX_COLORS[index % BOX_COLORS.length];
      item.querySelector(".confidence-bar span").style.width = `${percentage}%`;
      detectionList.append(item);
    });
  }

  function updatePredictionsForSettings() {
    if (!mediaStream || !model) return;
    latestPredictions = latestPredictions.filter(
      (prediction) => prediction.score >= Number(confidenceSlider.value) / 100
    );
    objectsFound.textContent = String(latestPredictions.length);
    detectionCount.textContent = String(latestPredictions.length);
    drawPredictions(latestPredictions);
    renderDetections(latestPredictions);
  }

  startButton.addEventListener("click", startCamera);
  stopButton.addEventListener("click", stopCamera);
  confidenceSlider.addEventListener("input", () => {
    updateSliderFill();
    updatePredictionsForSettings();
  });
  boxesToggle.addEventListener("change", () => drawPredictions(latestPredictions));
  labelsToggle.addEventListener("change", () => drawPredictions(latestPredictions));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && mediaStream) stopCamera();
  });
  document.getElementById("helpButton").addEventListener("click", () => {
    document.getElementById("about").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  window.addEventListener("beforeunload", () => {
    disposed = true;
    stopCamera();
  });

  updateSliderFill();
  loadModel();
})();
