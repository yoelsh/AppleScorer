# -*- coding: utf-8 -*-
"""
Rank dishes by the relative brightness to BIOHEKER.
Open multiple images at once. The darkest dish in all images gets 100. 
All other images are ranked relative to it.
You can move to the next and previos images.

Created on Sat May 24 18:26:38 2025

@author: yoelsh
"""

import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

# Store processed results globally
all_results = []
current_index = 0

def detect_circles_from_multiple(images):
    global all_results
    all_circles = []

    for path in images:
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=50, param2=30, minRadius=20, maxRadius=70)

        valid = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for (x, y, r) in circles:
                inner_mask = np.zeros_like(gray)
                outer_mask = np.zeros_like(gray)

                cv2.circle(inner_mask, (x, y), r, 255, -1)
                cv2.circle(outer_mask, (x, y), int(r * 1.3), 255, -1)
                cv2.circle(outer_mask, (x, y), r + 2, 0, -1)

                inner_brightness = cv2.mean(gray, mask=inner_mask)[0]
                background_brightness = cv2.mean(gray, mask=outer_mask)[0]

                if background_brightness > 100:
                    valid.append((x, y, r, inner_brightness))
                    all_circles.append((path, x, y, r, inner_brightness))

        all_results.append({'path': path, 'image': image_rgb, 'circles': valid})

    return all_circles

def process_and_score(images):
    global all_results
    all_results.clear()

    all_circles = detect_circles_from_multiple(images)
    if not all_circles:
        return

    darkest = min(c[-1] for c in all_circles)

    for result in all_results:
        img = result['image'].copy()
        scored = []

        for x, y, r, b in result['circles']:
            score = 100 * darkest / b if b != 0 else 0
            scored.append((x, y, r, b, score))
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            #cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
            cv2.putText(img, f"{score:.0f}", (x - 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        result['image_scored'] = img
        result['scored_circles'] = sorted(scored, key=lambda x: x[4], reverse=True)

def show_image(index):
    if index < 0 or index >= len(all_results):
        return

    result = all_results[index]
    img_pil = Image.fromarray(result['image_scored'])
    img_pil.thumbnail((800, 800))
    tk_img = ImageTk.PhotoImage(img_pil)

    image_label.config(image=tk_img)
    image_label.image = tk_img

    #name = result['path'].split('/')[-1]
    #top5 = result['scored_circles'][:5]
    #text = f"{name}\nTop 5 Circles:\n" + "\n".join(
    #    [f"({x},{y}) → Score {s:.1f}" for x, y, r, b, s in top5]
    #)
    #result_label.config(text=text)

def open_multiple_images():
    global current_index
    paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not paths:
        return
    process_and_score(paths)
    current_index = 0
    show_image(current_index)

def show_next():
    global current_index
    if current_index < len(all_results) - 1:
        current_index += 1
        show_image(current_index)

def show_prev():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image(current_index)

# GUI setup
root = Tk()
root.title("Petri Dish Multi-Image Scorer")

Button(root, text="Open Images", command=open_multiple_images).pack(pady=5)
Button(root, text="Previous Image", command=show_prev).pack()
Button(root, text="Next Image", command=show_next).pack()

image_label = Label(root)
image_label.pack()
result_label = Label(root, text="", justify="left", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
