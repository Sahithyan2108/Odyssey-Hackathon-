
import cv2
import os
from ultralytics import YOLO

# ==========================================
# YOLO ROAD ACCIDENT DEMONSTRATION
# Local vehicle detection + collision cues
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO loaded successfully!")

video_path = input("\nEnter your video path: ").strip().strip('"')

if not os.path.isfile(video_path):
    print("Video not found!")
    raise SystemExit

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video!")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

output_path = os.path.splitext(video_path)[0] + "_yolo_demo.mp4"

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    print("Could not create output video!")
    cap.release()
    raise SystemExit

# COCO vehicle classes:
# 2 = Car
# 3 = Motorcycle
# 5 = Bus
# 7 = Truck

VEHICLE_CLASSES = [2, 3, 5, 7]

overlap_frames = 0
WARNING_FRAMES = 8

print("\nStarting YOLO accident demonstration...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO locally on NVIDIA GPU
    results = model.predict(
        frame,
        conf=0.35,
        imgsz=640,
        classes=VEHICLE_CLASSES,
        device=0,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            label = model.names[class_id]

            detections.append((x1, y1, x2, y2))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Check for overlapping vehicle bounding boxes

    possible_collision = False

    for i in range(len(detections)):

        for j in range(i + 1, len(detections)):

            a = detections[i]
            b = detections[j]

            ix1 = max(a[0], b[0])
            iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2])
            iy2 = min(a[3], b[3])

            intersection = (
                max(0, ix2 - ix1) *
                max(0, iy2 - iy1)
            )

            area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
            area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))

            overlap = intersection / min(area_a, area_b)

            if overlap >= 0.25:
                possible_collision = True

    if possible_collision:
        overlap_frames += 1
    else:
        overlap_frames = 0

    # Display warning

    if overlap_frames >= WARNING_FRAMES:

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 65),
            (0, 0, 200),
            -1
        )

        cv2.putText(
            frame,
            "WARNING: POSSIBLE COLLISION - REVIEW VIDEO",
            (15, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "YOLO ROAD MONITORING ACTIVE",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Vehicles detected: {len(detections)}",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    cv2.imshow("YOLO Road Accident Demonstration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nAnalysis completed!")
print("Annotated video saved at:")
print(output_path)
