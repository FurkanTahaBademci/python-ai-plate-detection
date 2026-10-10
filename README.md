# python-ai-plate-detection

Detects license plates in a video with a custom-trained YOLOv4 model and reads the
characters with Tesseract OCR. Results are shown live in a PyQt5 desktop interface.

Yapay zeka ile tespit edilen plakaların görüntü işleme ve OCR ile metne dönüştürülmesi.

![demo](example/video_2022-03-29_16-28-51.gif)

## Pipeline

1. **Detection:** A YOLOv4 (Darknet) plate model runs through OpenCV's DNN module
   (`cv2.dnn_DetectionModel`, 416×416 input, confidence 0.6, NMS 0.4).
2. **Crop:** Each detected plate is cropped and resized to 300×80.
3. **OCR:** The HSV value (brightness) channel of the crop is passed to Tesseract.
   Readings between 6 and 9 characters are accepted and shown in the interface.
4. **Visualisation:** The interface shows the annotated frame, the plate crop, a
   preprocessed version (sharpening → median blur → Otsu threshold) and the FPS.

## Setup

1. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   The path is set in `main.py` (`pytesseract.pytesseract.tesseract_cmd`,
   default `C:\Program Files\Tesseract-OCR\tesseract.exe`); change it for your system.
2. Download the plate weights from
   [Google Drive](https://drive.google.com/file/d/10Xq44rII8Xt0YlhQKQDn5jL6cbGyjGc3/view?usp=sharing)
   and save them as `custom.weights` in the repository folder.
3. Put the input video in the folder as `video_trafik.mp4` (the file name is set in `main.py`).
4. Install the Python dependencies and run:

   ```
   pip install -r requirements.txt
   python main.py
   ```

Use the start/stop buttons in the window to run the video.

## Files

| File              | Description                         |
| ----------------- | ----------------------------------- |
| `main.py`         | Detection, OCR and interface logic  |
| `framework.py`    | PyQt5 window layout                 |
| `yolov4-obj.cfg`  | YOLOv4 network configuration        |
| `coco.names`      | Class names                         |
