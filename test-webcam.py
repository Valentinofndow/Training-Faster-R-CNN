import torch
import torchvision
from torchvision.transforms import transforms
import cv2
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 5  # background + 4 classes

# Build model
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

transform = transforms.ToTensor()
CONF_THRESHOLD = 0.7

# Open webcam (Windows: use CAP_DSHOW to avoid delay)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_time = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if needed (uncomment / adjust)
        # frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(rgb).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            preds = model(img_tensor)[0]
        t1 = time.time()

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        # Draw detections
        for box, lab, sc in zip(boxes, labels, scores):
            if sc < CONF_THRESHOLD:
                continue
            xmin, ymin, xmax, ymax = box.astype(int)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            class_name = CLASS_NAMES[lab - 1]
            label = f"{class_name}: {sc:.2f}"
            cv2.putText(frame, label, (xmin, max(15, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # FPS calculation (smoothed)
        curr_time = time.time()
        instant_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        fps = 0.9 * fps + 0.1 * instant_fps
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Optional: show inference time
        cv2.putText(frame, f"Infer: {(t1 - t0)*1000:.1f} ms", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        cv2.imshow("Real-time Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
