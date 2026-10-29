
document.addEventListener("DOMContentLoaded", () => {
  const $ = id => document.getElementById(id);

  const video = $("cameraPreview");
  const canvas = $("capturedFrame");
  const preview = $("capturedPreview");
  const upload = $("imageUpload");
  const startBtn = $("startCameraBtn");
  const captureBtn = $("captureFrameBtn");
  const detectBtn = $("detectApplianceBtn");
  const stopBtn = $("stopCameraBtn");
  const cameraStatus = $("cameraStatus");
  const detectionStatus = $("detectionStatus");
  const results = $("results");
  const sendBtn = $("sendDataBtn");
  const esp32Status = $("esp32Status");

  let stream = null;
  let selectedBlob = null;
  let previewUrl = null;
  let latestApplianceData = null;
  let detectionInProgress = false;

  const allowedAppliances = [
    "AC",
    "Air_Cooler",
    "Fan",
    "Light_Bulb",
    "Microwave",
    "TV",
    "Washing_Machine"
  ];

  function normalizeAppliance(value) {
    const label = String(value || "").trim();

    return allowedAppliances.find(
      appliance => appliance.toLowerCase() === label.toLowerCase()
    ) || null;
  }

  function setSendStatus(message) {
    if (esp32Status) {
      esp32Status.textContent = message;
    }
  }

  function clearPrediction(message = "Detect an appliance to enable data sending.") {
    latestApplianceData = null;
    sendBtn.disabled = true;

    if (typeof window.setLatestApplianceData === "function") {
      window.setLatestApplianceData(null);
    }

    setSendStatus(message);
  }

  function selectImage(blob, filename) {
    selectedBlob = blob;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(blob);
    preview.src = previewUrl;
    preview.hidden = false;

    detectBtn.disabled = false;
    results.hidden = true;

    clearPrediction("Image ready. Detect the appliance before sending data.");
    detectionStatus.textContent = `Image ready: ${filename}`;
  }

  startBtn.addEventListener("click", async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access requires localhost or HTTPS.");
      }

      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      cameraStatus.textContent = "Requesting camera permission...";

      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      video.srcObject = stream;
      video.hidden = false;
      await video.play();

      captureBtn.disabled = false;
      cameraStatus.textContent = "Camera started.";
    } catch (error) {
      cameraStatus.textContent = "Camera error: " + error.message;
    }
  });

  captureBtn.addEventListener("click", () => {
    if (!stream || !video.videoWidth) {
      cameraStatus.textContent = "Start the camera first.";
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      cameraStatus.textContent = "Could not access the capture canvas.";
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
      if (!blob) {
        cameraStatus.textContent = "Could not capture the image.";
        return;
      }

      upload.value = "";
      selectImage(blob, "camera-capture.jpg");
      cameraStatus.textContent = "Image captured.";
    }, "image/jpeg", 0.9);
  });

  upload.addEventListener("change", () => {
    const file = upload.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      cameraStatus.textContent = "Please select an image file.";
      upload.value = "";
      return;
    }

    selectImage(file, file.name);
    cameraStatus.textContent = "Image uploaded.";
  });

  detectBtn.addEventListener("click", async () => {
    if (!selectedBlob || detectionInProgress) {
      if (!selectedBlob) {
        detectionStatus.textContent = "Capture or upload an image first.";
      }
      return;
    }

    const form = new FormData();
    form.append("image", selectedBlob, "appliance-image.jpg");

    detectionInProgress = true;
    detectBtn.disabled = true;
    clearPrediction("Analyzing appliance image...");

    detectionStatus.textContent = "Analyzing image...";

    try {
      const response = await fetch("/api/detect-appliance", {
        method: "POST",
        body: form
      });

      const data = await response.json();

      if (!response.ok || data.success === false) {
        throw new Error(
          data.error || data.message || "Detection failed."
        );
      }

      const appliance = normalizeAppliance(
        data.name ?? data.appliance ?? data.class_name
      );

      const confidence = Number(data.confidence);
      const minWatts = Number(data.min_watts);
      const maxWatts = Number(data.max_watts);

      if (
        !appliance ||
        !Number.isFinite(confidence) ||
        confidence < 0 ||
        confidence > 100 ||
        !Number.isFinite(minWatts) ||
        !Number.isFinite(maxWatts) ||
        minWatts < 0 ||
        maxWatts < minWatts
      ) {
        throw new Error(
          "The server returned incomplete prediction data. " +
          "Check the appliance name, confidence and wattage fields."
        );
      }

      latestApplianceData = {
        appliance,
        confidence,
        min_watts: minWatts,
        max_watts: maxWatts,
        estimated_power:
          Number.isFinite(Number(data.estimated_power))
            ? Number(data.estimated_power)
            : (minWatts + maxWatts) / 2
      };

      results.hidden = false;

      $("applianceName").textContent = `Appliance: ${appliance}`;
      $("applianceConfidence").textContent =
        `Model score: ${confidence}%`;

      $("applianceWattage").textContent =
        `Estimated power: ${minWatts}–${maxWatts} W`;

      $("detectionTime").textContent =
        data.detection_time != null
          ? `Processing time: ${data.detection_time} seconds`
          : "";

      $("applianceMessage").textContent = data.message || "";

      // This is the key fix: enable the button after valid detection.
      sendBtn.disabled = false;

      // Make the same prediction available to other page scripts.
      window.setLatestApplianceData = window.setLatestApplianceData ||
        function (prediction) {
          window.latestApplianceData = prediction;
        };

      window.setLatestApplianceData(latestApplianceData);
      window.latestApplianceData = latestApplianceData;

      setSendStatus("Detection successful. Ready to send data to ESP32.");
      detectionStatus.textContent = "Detection completed.";
    } catch (error) {
      clearPrediction("Detection failed. Detect a valid appliance first.");
      detectionStatus.textContent = error.message;
    } finally {
      detectionInProgress = false;
      detectBtn.disabled = !selectedBlob;
    }
  });

  sendBtn.addEventListener("click", async () => {
    if (!latestApplianceData) {
      setSendStatus("Please detect an appliance before sending data.");
      sendBtn.disabled = true;
      return;
    }

    let esp32BaseUrl;

    try {
      const url = new URL($("esp32Url").value.trim());

      if (
        !["http:", "https:"].includes(url.protocol) ||
        url.username ||
        url.password
      ) {
        throw new Error("Invalid address");
      }

      esp32BaseUrl = url.origin;
    } catch {
      setSendStatus("Enter a valid ESP32 URL, such as http://192.168.4.1.");
      return;
    }

    sendBtn.disabled = true;
    setSendStatus("Sending appliance data...");

    try {
      const response = await fetch("/api/send-to-esp32", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          esp32_url: esp32BaseUrl,
          appliance: latestApplianceData.appliance,
          confidence: latestApplianceData.confidence,
          min_watts: latestApplianceData.min_watts,
          max_watts: latestApplianceData.max_watts,
          estimated_power: latestApplianceData.estimated_power
        })
      });

      const result = await response.json().catch(() => ({}));

      if (!response.ok || result.success !== true) {
        throw new Error(
          result.message || result.error || "The server could not send the data."
        );
      }

      setSendStatus("Success: appliance data sent to ESP32.");
    } catch (error) {
      setSendStatus(
        "Send failed: " + error.message +
        " Check your Flask and ESP32 connection."
      );
    } finally {
      sendBtn.disabled = !latestApplianceData;
    }
  });

  stopBtn.addEventListener("click", () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }

    stream = null;
    video.srcObject = null;
    video.hidden = true;
    captureBtn.disabled = true;

    cameraStatus.textContent = "Camera stopped.";
  });

  window.getLatestApplianceData = () => latestApplianceData;
});
