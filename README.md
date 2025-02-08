# Car Detection in Video using OpenCV

This project uses OpenCV's Haar Cascade Classifier to detect cars in a video, draw bounding boxes around them, and save the processed video as output. It is a great example of real-time object detection and video processing.

---

## Features
- Detects cars in a video using a pre-trained Haar Cascade Classifier.
- Processes video frames and highlights detected cars with bounding boxes.
- Saves the processed video with detections to an output file.

---

## Demo
Watch the live demo of the project on YouTube:
[Car Detection Demo](https://www.youtube.com/watch?v=mS23DrigsUU)

---

## Requirements
Ensure you have the following installed:
- Python 3.x
- OpenCV

Install the required dependencies with:
```bash
pip install opencv-python
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/HariPasapuleti/Car-detection.git
   cd Car-detection
   ```

2. Add the Haar Cascade XML file (`cars.xml`) to the project directory. You can download it from [OpenCV GitHub](https://github.com/opencv/opencv).

3. Place your input video in the `videos/` directory and update the `video_source` variable in the script.

4. Run the script:
   ```bash
   python carDetection.py
   ```

5. The processed video will be saved as `output.mp4` in the project directory.

---

## Code Overview
- **`carDetection.py`**: Main script for processing video and detecting cars.
- **`cars.xml`**: Haar Cascade file for car detection.
- **`videos/`**: Directory to store input video files.

---

## Output
- Bounding boxes drawn around detected cars in each video frame.
- Saved video (`output.mp4`) with detections included.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contribution
Contributions are welcome! Feel free to open issues or submit pull requests for enhancements or bug fixes.
