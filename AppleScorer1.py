# -*- coding: utf-8 -*-
"""
Rank dishes by the relative brightness to BIOHEKER.
Each image is ranked independently.

Created on Sat May 24 18:07:06 2025

@author: yoelsh
"""

import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

def detect_valid_circles(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 5)

    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=70)

    output_image = image_rgb.copy()
    valid_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))

        for i, (x, y, r) in enumerate(circles):
            inner_mask = np.zeros_like(gray)
            outer_mask = np.zeros_like(gray)

            cv2.circle(inner_mask, (x, y), r, 255, -1)
            cv2.circle(outer_mask, (x, y), int(r * 1.3), 255, -1)
            cv2.circle(outer_mask, (x, y), r + 2, 0, -1)

            inner_brightness = cv2.mean(gray, mask=inner_mask)[0]
            background_brightness = cv2.mean(gray, mask=outer_mask)[0]

            if background_brightness > 100:
                valid_circles.append((x, y, r, inner_brightness))

        if valid_circles:
            # Find darkest brightness
            darkest = min(circle[3] for circle in valid_circles)

            scored_circles = []
            for x, y, r, brightness in valid_circles:
                score = 100 * darkest / brightness if brightness != 0 else 0
                scored_circles.append((x, y, r, brightness, score))

                # Draw and annotate
                cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
                #cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(output_image, f"{score:.0f}", (x - 10, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            scored_circles.sort(key=lambda x: x[4], reverse=True)
            return output_image, scored_circles

    return output_image, []

def show_results():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    result_img, scored = detect_valid_circles(file_path)

    result_img_pil = Image.fromarray(result_img)
    result_img_pil.thumbnail((800, 800))
    tk_image = ImageTk.PhotoImage(result_img_pil)

    image_label.config(image=tk_image)
    image_label.image = tk_image

    top5 = scored[:5]
    #text = "\n".join([f"Circle at ({x}, {y}): Score {score:.1f}" for x, y, r, b, score in top5])
    #result_label.config(text=f"Top 5 Darkest Circles (Score out of 100):\n{text}")

# GUI setup
root = Tk()
root.title("Petri Dish Darkness Scorer")

Button(root, text="Open Image", command=show_results).pack(pady=10)
image_label = Label(root)
image_label.pack()
result_label = Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
