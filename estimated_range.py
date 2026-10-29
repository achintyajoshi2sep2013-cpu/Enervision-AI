# ENERVISION-AI: Image-Based Power Consumption Estimator

import json
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "database" / "enervision_appliances.keras"
CLASSES_PATH = BASE_DIR / "database" / "class_names.json"

APPLIANCE_POWER_RANGES = {
    "AC": (800, 2500),
    "Air_Cooler": (100, 300),
    "Fan": (30, 80),
    "Light_Bulb": (5, 20),
    "Microwave": (600, 1500),
    "TV": (40, 150),
    "Washing_Machine": (400, 2000),
}

model = None
class_names = []


def load_model():
    global model, class_names

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        messagebox.showerror(
            "Files Missing",
            "Check that these files exist:\n"
            "database/enervision_appliances.keras\n"
            "database/class_names.json"
        )
        return False

    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)

        model = tf.keras.models.load_model(MODEL_PATH)
        return True

    except Exception as error:
        messagebox.showerror("Loading Error", str(error))
        return False


def choose_image():
    file_path = filedialog.askopenfilename(
        title="Choose an Appliance Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        with Image.open(file_path) as image:
            image = image.convert("RGB")

            input_shape = model.input_shape
            height = input_shape[1] or 160
            width = input_shape[2] or 160

            resized = image.resize((width, height))
            image_array = np.asarray(resized, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            if len(predictions) != len(class_names):
                raise ValueError(
                    "Model outputs and class_names.json do not match."
                )

            best_index = int(np.argmax(predictions))
            appliance = class_names[best_index]
            confidence = float(predictions[best_index]) * 100

            # Show the chosen image.
            preview = Image.open(file_path).convert("RGB")
            preview.thumbnail((350, 240))
            photo = ImageTk.PhotoImage(preview)

            image_label.configure(image=photo, text="")
            image_label.image = photo

            if appliance not in APPLIANCE_POWER_RANGES:
                result_label.configure(
                    text=f"Identified: {appliance}\n"
                         "No power range is configured for this class."
                )
                return

            minimum, maximum = APPLIANCE_POWER_RANGES[appliance]

            result_label.configure(
                text=(
                    f"Appliance: {appliance.replace('_', ' ')}\n"
                    f"Model Score: {confidence:.2f}%\n\n"
                    f"Estimated Power: {minimum}–{maximum} W\n"
                    f"Estimated Power: {minimum / 1000:.2f}–"
                    f"{maximum / 1000:.2f} kW\n\n"
                    f"Estimated Energy for 1 hour:\n"
                    f"{minimum / 1000:.2f}–"
                    f"{maximum / 1000:.2f} kWh"
                )
            )

            path_label.configure(text=Path(file_path).name)

    except Exception as error:
        messagebox.showerror("Prediction Error", str(error))


root = tk.Tk()
root.title("ENERVISION-AI | Power Estimator")
root.geometry("560x570")
root.configure(bg="#101827")

tk.Label(
    root,
    text="ENERVISION-AI",
    font=("Arial", 23, "bold"),
    fg="#38BDF8",
    bg="#101827"
).pack(pady=(20, 5))

tk.Label(
    root,
    text="Appliance Recognition & Energy Estimation",
    font=("Arial", 11),
    fg="white",
    bg="#101827"
).pack(pady=(0, 15))

tk.Button(
    root,
    text="Choose an Image",
    command=choose_image,
    font=("Arial", 12, "bold"),
    bg="#0284C7",
    fg="white",
    activebackground="#0369A1",
    padx=18,
    pady=10
).pack(pady=10)

path_label = tk.Label(
    root,
    text="No image selected",
    font=("Arial", 10),
    fg="#CBD5E1",
    bg="#101827"
)
path_label.pack(pady=5)

image_label = tk.Label(root, bg="#101827")
image_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="Choose an appliance image to begin.",
    font=("Arial", 12, "bold"),
    fg="#67E8F9",
    bg="#101827",
    justify="center",
    wraplength=510
)
result_label.pack(pady=15)

if load_model():
    root.mainloop()
else:
    root.destroy()