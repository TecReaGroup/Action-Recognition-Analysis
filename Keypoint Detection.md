# Keypoint  Detection

| 排名 | 算法                               | 精度                 | PyTorch FP16     | TensorRT FP16    |
| ---- | ---------------------------------- | -------------------- | ---------------- | ---------------- |
| 1    | ViTPose-B（每个框单独估点）        | 最高                 | 每框约 18–28 FPS | 每框约 35–50 FPS |
| 2    | RF-DETR Keypoint（576）            | 高                   | 约 18–28 FPS     | 约 45–65 FPS     |
| 3    | RTMPose-m/l + 检测器（1–3 个物体） | 高                   | 约 30–50 FPS     | 约 70–110 FPS    |
| 4    | YOLO-pose-l                        | 中高                 | 约 28–36 FPS     | 约 55–80 FPS     |
| 5    | RTMO-l                             | 中高                 | 约 22–35 FPS     | 约 50–80 FPS     |
| 6    | YOLO-pose-m                        | 中（最推荐先训这个） | 约 45–55 FPS     | 约 90–130 FPS    |
| 7    | ED-Pose R50                        | 中                   | 约 10–16 FPS     | 约 20–32 FPS     |
| 8    | YOLO-pose-s                        | 中下                 | 约 85–105 FPS    | 约 160–220 FPS   |
| 9    | Keypoint R-CNN R50                 | 中下                 | 约 8–14 FPS      | 约 15–22 FPS     |
| 10   | YOLO-pose-n                        | 低                   | 约 150–180 FPS   | 约 280–350 FPS   |