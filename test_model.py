# ENERVISION-AI: Choose an Image and Identify the Appliance

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

model = None
class_names = []


def load_model():
    global model, class_names

    if not MODEL_PATH.exists():
        messagebox.showerror(
            "Model Not Found",
            f"Could not find the model:\n{MODEL_PATH}"
        )
        return False

    if not CLASSES_PATH.exists():
        messagebox.showerror(
            "Classes Not Found",
            f"Could not find the class list:\n{CLASSES_PATH}"
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

            # Read the model's required image dimensions.
            input_shape = model.input_shape
            height = input_shape[1] or 160
            width = input_shape[2] or 160

            resized = image.resize((width, height))
            image_array = np.asarray(resized, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # Do not normalize again if the saved model
            # already contains a Rescaling layer.
            predictions = model.predict(image_array, verbose=0)[0]

            if len(predictions) != len(class_names):
                raise ValueError(
                    "The model output count does not match "
                    "the number of class names."
                )

            best_index = int(np.argmax(predictions))
            appliance = class_names[best_index]
            confidence = float(predictions[best_index]) * 100

            # Display the selected image.
            preview = Image.open(file_path).convert("RGB")
            preview.thumbnail((420, 300))
            photo = ImageTk.PhotoImage(preview)

            image_label.configure(image=photo, text="")
            image_label.image = photo

            result_label.configure(
                text=(
                    f"Predicted Appliance: {appliance.replace('_', ' ')}\n"
                    f"Model Score: {confidence:.2f}%"
                )
            )

            path_label.configure(text=Path(file_path).name)

    except Exception as error:
        messagebox.showerror("Prediction Error", str(error))


root = tk.Tk()
root.title("ENERVISION-AI | Appliance Recognition")
root.geometry("560x540")
root.configure(bg="#101827")

title = tk.Label(
    root,
    text="ENERVISION-AI",
    font=("Arial", 23, "bold"),
    fg="#38BDF8",
    bg="#101827"
)
title.pack(pady=(20, 5))

subtitle = tk.Label(
    root,
    text="AI Appliance Recognition",
    font=("Arial", 12),
    fg="white",
    bg="#101827"
)
subtitle.pack(pady=(0, 15))

choose_button = tk.Button(
    root,
    text="Choose an Image",
    command=choose_image,
    font=("Arial", 12, "bold"),
    bg="#0284C7",
    fg="white",
    activebackground="#0369A1",
    padx=18,
    pady=10
)
choose_button.pack(pady=10)

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
    font=("Arial", 14, "bold"),
    fg="#67E8F9",
    bg="#101827",
    justify="center"
)
result_label.pack(pady=15)

if load_model():
    root.mainloop()
else:
    root.destroy()