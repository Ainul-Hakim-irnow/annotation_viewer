import os
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.widgets import Button

class ImageViewer:
    def __init__(self, image_folder, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_folder = image_folder
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.data['categories']}
        self.current_index = 0
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)

        self.button_next = Button(plt.axes([0.8, 0.05, 0.1, 0.075]), 'Next')
        self.button_prev = Button(plt.axes([0.1, 0.05, 0.1, 0.075]), 'Previous')
        self.button_next.on_clicked(self.next_image)
        self.button_prev.on_clicked(self.prev_image)

        self.show_image()

    def show_image(self):
        self.ax.clear()

        image_data = self.images[self.current_index]
        image_path = os.path.join(self.image_folder, image_data['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.ax.imshow(image)
        self.ax.set_title(f"{image_data['file_name']} ({self.current_index + 1}/{len(self.images)})")

        annotations = [anno for anno in self.annotations if anno['image_id'] == image_data['id']]

        for anno in annotations:
            bbox = anno['bbox']
            segmentation = anno['segmentation']
            category_id = anno['category_id']
            category_name = self.categories.get(category_id, 'Unknown')

            # Draw bounding box
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)
            self.ax.text(bbox[0], bbox[1] - 5, category_name, color='red', fontsize=10, backgroundcolor='white')

            # Draw segmentation
            if isinstance(segmentation, list):  # Check for polygon format
                for seg in segmentation:
                    if len(seg) >= 6:  # Ensure enough points to form a polygon
                        reshaped_seg = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                        polygon = Polygon(reshaped_seg, linewidth=1, edgecolor='blue', facecolor='none')
                        self.ax.add_patch(polygon)

        plt.draw()

    def next_image(self, event):
        self.current_index = (self.current_index + 1) % len(self.images)
        self.show_image()

    def prev_image(self, event):
        self.current_index = (self.current_index - 1) % len(self.images)
        self.show_image()

# Replace these paths with your actual folder and JSON file
image_folder = "/home/ainul/Documents/Datasets/Bank Note.v1i.coco/train"
json_path = "/home/ainul/Documents/Datasets/Bank Note.v1i.coco/train/_annotations.coco.json"

# image_folder = "/home/ainul/Documents/Datasets/Detergent Bag.v31i.coco/aug"
# json_path = "/home/ainul/Documents/Datasets/Detergent Bag.v31i.coco/aug/_annotations.coco.json"

viewer = ImageViewer(image_folder, json_path)
plt.show()
