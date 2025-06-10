# Vehicle Speed Tracking with Homography and Manual Object Selection

This project demonstrates a simple but complete computer vision pipeline for estimating vehicle speed using a video input. It includes object detection with YOLOv5, manual object selection, speed estimation, and homography-based top-down mapping.

## 📦 Features

- **YOLOv5 detection** of vehicles (car, truck, bus, etc.)
- **Manual vehicle selection** at the start via mouse click
- **Speed estimation** based on pixel displacement and calibrated scale
- **Homography transformation** to display the movement on a 2D map
- **Dual display**: video with overlay + live top-down map

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Additionally, make sure you have PyTorch and a compatible version of YOLOv5 via `torch.hub`.

## 📂 Files

- `main.py`: Main script to run the full system
- `detect_one_car.py`: Vehicle detection and manual selection logic
- `estimate_speed.py`: Speed calculation function
- `homography_map.py`: Homography selection and coordinate transformation
- `requirements.txt`: Python dependencies

## 🎮 How to Use

1. Place your input video as `traffic.mp4` in the project folder.
2. Extract a frame and resize it to match processing scale (`0.3`) and save it as `Frame.jpg`.
3. Run the following to define your homography:

```bash
python get_homography.py
```

4. Run the main script:

```bash
python main.py
```

Click on the vehicle you want to track. The system will follow it, estimate its speed, and show a point on the map.

## 📈 Output

- Live video with bounding box and estimated speed (in km/h)
- Top-down map showing the vehicle's movement using homography

## 📌 Notes

- This project tracks **only one vehicle** at a time using manual selection.
- YOLOv5 is loaded via `torch.hub`, no training needed.
- Scale (`meters_per_pixel`) must be calibrated beforehand for real-world units.

## 📷 Sample

*(Add a screenshot or short video demo here)*

## 📄 License

MIT License