# -*- coding: utf-8 -*-
"""
Like AppleScorer2, but show all images at the same time and add a save button.

Created on Sat May 24 18:31:22 2025

@author: yoelsh
"""

import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog, Canvas, Frame, Scrollbar, VERTICAL, RIGHT, LEFT, BOTH, Y
from PIL import Image, ImageTk

from datetime import datetime

all_results = []

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

def show_all_images():
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    pil_images = []

    for result in all_results:
        img_pil = Image.fromarray(result['image_scored'])
        img_pil.thumbnail((800, 800))
        pil_images.append(img_pil)

        tk_img = ImageTk.PhotoImage(img_pil)
        lbl = Label(scrollable_frame, image=tk_img)
        lbl.image = tk_img  # keep reference
        lbl.pack(pady=10)

        #name = result['path'].split("/")[-1]
        #top5 = result['scored_circles'][:5]
        #text = f"{name}\nTop 5 Circles:\n" + "\n".join(
        #    [f"({x},{y}) → Score {s:.1f}" for x, y, r, b, s in top5]
        #)
        #Label(scrollable_frame, text=text, font=("Arial", 12), justify="left").pack()

    return pil_images

def open_images():
    paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not paths:
        return
    process_and_score(paths)
    show_all_images()

def save_combined_image():
    pil_images = [Image.fromarray(r['image_scored']) for r in all_results]
    widths, heights = zip(*(img.size for img in pil_images))
    max_width = max(widths)
    total_height = sum(heights)

    stitched = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in pil_images:
        stitched.paste(img, (0, y_offset))
        y_offset += img.size[1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stitched.save(f"summary_{timestamp}.jpg")
    print(f"Saved: summary_{timestamp}.jpg")

# GUI
root = Tk()
root.title("Petri Dish Multi-Image Scorer")

Button(root, text="Open Images", command=open_images).pack(pady=5)
Button(root, text="Save Results as JPG", command=save_combined_image).pack(pady=5)

# Scrollable canvas
canvas = Canvas(root)
scrollbar = Scrollbar(root, orient=VERTICAL, command=canvas.yview)
scrollable_frame = Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

root.mainloop()
