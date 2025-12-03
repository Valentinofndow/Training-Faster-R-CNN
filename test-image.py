import torch
import torchvision
from torchvision.transforms import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5  # background + 4 classes

# Build model again
# Model 1 2 3
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="COCO_V1")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)

# Load trained weights
model.load_state_dict(torch.load("model/model4.pth", map_location=device))

model.to(device)
model.eval()

# ------- CLASS NAME MAPPING (IMPORTANT) -------
CLASS_NAMES = [
    'Double_Print',
    'Missing_Text',
    'Normal',
    'Touching_Characters'
]
# ------------------------------------------------
start_time = time.time()
# Path gambar yang mau diuji
image_path = "test/1.jpg"

# Load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = transforms.ToTensor()
img_tensor = transform(image_rgb).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    preds = model(img_tensor)[0]

boxes = preds["boxes"].cpu().numpy()
labels = preds["labels"].cpu().numpy()
scores = preds["scores"].cpu().numpy()

plt.figure(figsize=(10, 8))
plt.imshow(image_rgb)
ax = plt.gca()

for box, lab, sc in zip(boxes, labels, scores):
    if sc < 0.7:
        continue

    xmin, ymin, xmax, ymax = box

    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    # Convert model label → YOLO label → class name
    class_name = CLASS_NAMES[lab - 1]

    ax.text(
        xmin,
        ymin - 5,
        f"{class_name} ({sc:.2f})",
        fontsize=12,
        color="yellow",
        bbox=dict(facecolor="red", alpha=0.4)
    )
end_time = time.time()
print(f"Inference Time: {end_time - start_time:.3f} seconds")
plt.show()
