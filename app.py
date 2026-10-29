
import os
import json
import time
import socket
import logging
import ipaddress
import subprocess
from io import BytesIO
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"

MODEL_PATH = DATABASE_DIR / "enervision_appliances.keras"
CLASS_NAMES_PATH = DATABASE_DIR / "class_names.json"

IMAGE_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.70
ESP32_DEFAULT_URL = "http://192.168.4.1"
ESP32_TIMEOUT = 5
MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Set ENERVISION_PREPROCESSING in your environment if necessary:
# mobilenet_v2 = use preprocess_input()
# raw           = keep pixel values between 0 and 255
# auto           = use raw pixels when a Rescaling layer is detected;
#                  otherwise use MobileNetV2 preprocessing.
PREPROCESSING_MODE = os.getenv(
    "ENERVISION_PREPROCESSING", "auto"
).strip().lower()

ALLOWED_APPLIANCES = {
    "AC",
    "Air_Cooler",
    "Fan",
    "Light_Bulb",
    "Microwave",
    "TV",
    "Washing_Machine",
}

POWER_RANGES = {
    "AC": (800, 2500),
    "Air_Cooler": (100, 300),
    "Fan": (30, 80),
    "Light_Bulb": (5, 20),
    "Microwave": (600, 1500),
    "TV": (40, 150),
    "Washing_Machine": (400, 2000),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnervisionAI")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


# ============================================================
# LOAD CLASS LABELS
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    raw_class_names = json.load(file)

if isinstance(raw_class_names, list):
    class_names = [str(value).strip() for value in raw_class_names]

elif isinstance(raw_class_names, dict):
    try:
        class_names = [
            str(value).strip()
            for _, value in sorted(
                raw_class_names.items(),
                key=lambda item: int(item[0]),
            )
        ]
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "class_names.json must use a list or numeric dictionary keys."
        ) from exc

else:
    raise ValueError(
        "Invalid class_names.json format. Use a list or indexed dictionary."
    )

if len(class_names) != 7 or set(class_names) != ALLOWED_APPLIANCES:
    raise ValueError(
        "class_names.json must contain exactly these seven classes: "
        + ", ".join(sorted(ALLOWED_APPLIANCES))
        + ". Keep them in the exact order used during training."
    )


# ============================================================
# LOAD MODEL
# ============================================================

logger.info("Loading model from %s", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)

output_shape = model.output_shape

if isinstance(output_shape, list):
    raise ValueError("Only a single-output classification model is supported.")

if output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model output count ({output_shape[-1]}) does not match "
        f"class_names.json ({len(class_names)})."
    )


# ============================================================
# DETECT MODEL PREPROCESSING
# ============================================================

def has_rescaling_layer(loaded_model):
    """Check whether the model includes a Keras Rescaling layer."""
    visited = set()

    def inspect_layers(layers):
        for layer in layers:
            if id(layer) in visited:
                continue

            visited.add(id(layer))

            if layer.__class__.__name__ == "Rescaling":
                return True

            nested_layers = getattr(layer, "layers", None)
            if nested_layers and inspect_layers(nested_layers):
                return True

        return False

    return inspect_layers(getattr(loaded_model, "layers", []))


if PREPROCESSING_MODE not in {
    "auto", "raw", "mobilenet_v2"
}:
    raise ValueError(
        "ENERVISION_PREPROCESSING must be auto, raw, or mobilenet_v2."
    )

if PREPROCESSING_MODE == "auto":
    if has_rescaling_layer(model):
        effective_preprocessing = "raw"
    else:
        effective_preprocessing = "mobilenet_v2"
else:
    effective_preprocessing = PREPROCESSING_MODE

logger.info("Loaded classes in model output order: %s", class_names)
logger.info("Image preprocessing mode: %s", effective_preprocessing)


# ============================================================
# HELPERS
# ============================================================

def json_error(message, status=400):
    return jsonify({
        "success": False,
        "error": message,
        "message": message,
    }), status


def normalize_appliance(value):
    if not isinstance(value, str):
        return None

    value = value.strip().replace("-", "_").replace(" ", "_")

    for appliance in ALLOWED_APPLIANCES:
        if appliance.casefold() == value.casefold():
            return appliance

    return None


def prepare_image(image):
    """Prepare the image using the configured preprocessing mode."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32)

    if effective_preprocessing == "mobilenet_v2":
        array = preprocess_input(array)

    return np.expand_dims(array, axis=0)


def read_uploaded_image():
    """Accept a multipart image upload or a JSON image data URL."""
    if "image" in request.files:
        uploaded = request.files["image"]

        if not uploaded.filename:
            raise ValueError("Please select an image.")

        raw = uploaded.read(MAX_IMAGE_BYTES + 1)

    elif request.is_json:
        import base64
        import binascii

        data = request.get_json(silent=True) or {}
        image_data = data.get("image")

        if not isinstance(image_data, str) or "," not in image_data:
            raise ValueError("No valid image was supplied.")

        header, encoded = image_data.split(",", 1)

        if not header.startswith("data:image/"):
            raise ValueError("The supplied data is not an image.")

        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid image encoding.") from exc

    else:
        raise ValueError("Please upload or capture an image.")

    if not raw:
        raise ValueError("The uploaded image is empty.")

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("The image exceeds the 10 MB size limit.")

    try:
        with Image.open(BytesIO(raw)) as image:
            image.verify()

        return Image.open(BytesIO(raw)).convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


# ============================================================
# PREDICTION
# ============================================================

def predict_appliance(image):
    image_array = prepare_image(image)

    started = time.perf_counter()
    raw_output = model.predict(image_array, verbose=0)
    elapsed = time.perf_counter() - started

    scores = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    if len(scores) != len(class_names):
        raise ValueError("Model output does not match the seven class labels.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid prediction scores.")

    # This model is expected to return softmax probabilities.
    # Do not silently treat logits as probabilities.
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(np.sum(scores), 1.0, atol=0.02)
    ):
        raise ValueError(
            "The model output is not a valid probability distribution. "
            "Check the final layer of the trained model."
        )

    best_index = int(np.argmax(scores))
    appliance = class_names[best_index]
    confidence = float(scores[best_index]) * 100

    min_watts, max_watts = POWER_RANGES[appliance]

    if confidence >= CONFIDENCE_THRESHOLD * 100:
        message = "Appliance detected successfully."
    else:
        message = (
            "Low-confidence result. Try a clearer image with one "
            "appliance visible."
        )

    return {
        "success": True,
        "name": appliance,
        "appliance": appliance,
        "predicted_class": appliance,
        "confidence": round(confidence, 2),
        "confident_prediction": (
            confidence >= CONFIDENCE_THRESHOLD * 100
        ),
        "min_watts": min_watts,
        "max_watts": max_watts,
        "estimated_power": round((min_watts + max_watts) / 2, 1),
        "detection_time": round(elapsed, 3),
        "preprocessing": effective_preprocessing,
        "scores": [
            {
                "name": class_names[index],
                "score": round(float(scores[index]) * 100, 2),
            }
            for index in range(len(class_names))
        ],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "message": message,
    }


# ============================================================
# ESP32 URL VALIDATION
# ============================================================

def validate_esp32_url(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter the ESP32 address.")

    value = value.strip()

    if "://" not in value:
        value = "http://" + value

    parsed = urlparse(value)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Use an HTTP or HTTPS address.")

    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid ESP32 URL.")

    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("Enter the base URL without an additional path.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid port in ESP32 URL.") from exc

    if port not in (None, 80, 443, 5000, 8080):
        raise ValueError("Unsupported ESP32 port.")

    hostname = parsed.hostname

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.gethostbyname(hostname))
        except (socket.gaierror, ValueError) as exc:
            raise ValueError("Could not resolve the ESP32 address.") from exc

    if not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
    ):
        raise ValueError("Use a local network address for the ESP32.")

    return value.rstrip("/")


# ============================================================
# WEBSITE AND HEALTH
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "running",
        "model_loaded": True,
        "classes": class_names,
        "preprocessing": effective_preprocessing,
    })


# ============================================================
# DETECTION ENDPOINTS
# ============================================================

@app.route("/api/detect-appliance", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def detect_appliance():
    try:
        image = read_uploaded_image()
        result = predict_appliance(image)
        return jsonify(result)

    except ValueError as exc:
        return json_error(str(exc), 400)

    except Exception:
        logger.exception("Appliance detection failed")
        return json_error(
            "Detection failed. Check the Flask terminal for details.",
            500,
        )


# ============================================================
# WI-FI SCANNING ON WINDOWS
# ============================================================

@app.route("/api/wifi-networks", methods=["GET"])
def wifi_networks():
    if os.name != "nt":
        return jsonify({
            "success": False,
            "networks": [],
            "message": "Wi-Fi scanning through this endpoint requires Windows.",
        }), 501

    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            return jsonify({
                "success": False,
                "networks": [],
                "message": result.stderr.strip()
                or "Windows could not scan Wi-Fi networks.",
            }), 500

        networks = []
        seen = set()

        for line in result.stdout.splitlines():
            line = line.strip()

            if line.lower().startswith("ssid ") and ":" in line:
                ssid = line.split(":", 1)[1].strip()

                if ssid and ssid not in seen:
                    seen.add(ssid)
                    networks.append({"ssid": ssid})

        return jsonify({
            "success": True,
            "networks": networks,
            "message": "Wi-Fi scan completed.",
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "networks": [],
            "message": "Wi-Fi scanning timed out.",
        }), 504

    except Exception:
        logger.exception("Wi-Fi scanning failed")
        return jsonify({
            "success": False,
            "networks": [],
            "message": "Could not scan Wi-Fi networks.",
        }), 500


# ============================================================
# ESP32 CONNECTION CHECK
# ============================================================

@app.route("/api/connection-request", methods=["POST"])
def connection_request():
    data = request.get_json(silent=True) or {}

    try:
        esp32_url = validate_esp32_url(
            data.get("esp32_url") or ESP32_DEFAULT_URL
        )

        response = requests.get(
            f"{esp32_url}/data",
            timeout=ESP32_TIMEOUT,
        )

        if not response.ok:
            return json_error(
                f"ESP32 returned HTTP {response.status_code}.",
                502,
            )

        try:
            sensor_data = response.json()
        except ValueError:
            return json_error("ESP32 did not return valid JSON.", 502)

        return jsonify({
            "success": True,
            "message": "ESP32 connection successful.",
            "sensor_data": sensor_data,
        })

    except ValueError as exc:
        return json_error(str(exc))

    except requests.Timeout:
        return json_error("ESP32 connection timed out.", 504)

    except requests.RequestException:
        logger.exception("ESP32 connection check failed")
        return json_error(
            "Could not reach the ESP32. Check the Wi-Fi connection.",
            502,
        )


# ============================================================
# SEND APPLIANCE DATA TO ESP32
# ============================================================

@app.route("/api/send-to-esp32", methods=["POST"])
def send_to_esp32():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error("A JSON request body is required.")

    try:
        esp32_url = validate_esp32_url(
            data.get("esp32_url") or ESP32_DEFAULT_URL
        )
    except ValueError as exc:
        return json_error(str(exc))

    appliance = normalize_appliance(
        data.get("appliance") or data.get("name")
    )

    if appliance is None:
        return json_error("Unsupported appliance class.")

    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return json_error("Confidence must be a number.")

    if not np.isfinite(confidence) or not 0 <= confidence <= 100:
        return json_error("Confidence must be between 0 and 100.")

    min_watts, max_watts = POWER_RANGES[appliance]

    payload = {
        "source": "Enervision AI",
        "appliance": appliance,
        "confidence": round(confidence, 2),
        "min_watts": min_watts,
        "max_watts": max_watts,
        "estimated_power": round((min_watts + max_watts) / 2, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        response = requests.post(
            f"{esp32_url}/api/appliance-data",
            json=payload,
            timeout=ESP32_TIMEOUT,
        )

        if not response.ok:
            return json_error(
                f"ESP32 returned HTTP {response.status_code}. "
                "Check that the firmware implements /api/appliance-data.",
                502,
            )

        try:
            esp32_response = response.json()
        except ValueError:
            esp32_response = {}

        if esp32_response.get("success") is False:
            return json_error(
                esp32_response.get("message", "ESP32 rejected the data."),
                502,
            )

        return jsonify({
            "success": True,
            "message": "Appliance data sent to ESP32 successfully.",
            "sent_data": payload,
            "esp32_response": esp32_response,
        })

    except requests.Timeout:
        return json_error("ESP32 did not respond in time.", 504)

    except requests.RequestException:
        logger.exception("Failed to send data to ESP32")
        return json_error(
            "Could not reach the ESP32. Check its Wi-Fi connection.",
            502,
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("\n========================================")
    print("          ENERVISION AI")
    print("========================================")
    print("Website: http://127.0.0.1:5000")
    print("Default ESP32: http://192.168.4.1")
    print("Class labels:", class_names)
    print("Preprocessing:", effective_preprocessing)
    print("========================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
