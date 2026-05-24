import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "input_videos/input.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

max_frames = int(fps * 60)   # 1 minute

out = cv2.VideoWriter(
    "output_videos/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    player_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls != 0 or conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_w = x2 - x1
        box_h = y2 - y1

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Ignore huge foreground people
        if box_h > height * 0.75:
            continue

        # Ignore audience (too high in frame)
        if center_y < height * 0.35:
            continue

        # Keep only players near court center
        if center_x < width * 0.2 or center_x > width * 0.8:
            continue

        player_boxes.append((x1, y1, x2, y2))

    # choose 2 biggest filtered players
    player_boxes = sorted(
        player_boxes,
        key=lambda b: (b[2]-b[0]) * (b[3]-b[1]),
        reverse=True
    )[:2]

    # Draw only Player 1 and Player 2
    for i, (x1, y1, x2, y2) in enumerate(player_boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

        cv2.putText(
            frame,
            f"Player {i+1}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    out.write(frame)

    frame_count += 1
    if frame_count >= max_frames:
        break

cap.release()
out.release()

print("Done! Better player tracking video saved.")