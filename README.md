#  Computer Vision Tracking Project

This project demonstrates object tracking using the YOLOv5 model. The tracking process is implemented in the `main.py` file and uses videos as input.  
You can select which person to track by clicking on them in the video — the system is capable of tracking **any person** that appears in the input video.

---

## Project Structure

```
project_cv/
│
├── data/
│   ├── raw/             # Place your input videos here
│   └── processed/       # Output videos will be saved here after tracking
│
├── models/
│   └── yolov5su.pt      # YOLOv5 model used for tracking
│
├── src/
│   ├── main.py          # Main script to run tracking
│   ├── tracker.py       # Tracking logic
│   ├── utils.py         # Utility functions
│   └── visualization.py # Visualization helpers
│
├── req.txt              # Python dependencies
└── note.ipynb           # Optional notebook for testing
```

---

##  How to Run

1. **Clone the repository**:

```bash
git clone https://github.com/rahmaeBrahi/computer-vision-project.git
cd computer-vision-project
```

2. **Install required packages**:

```bash
pip install -r req.txt
```

3. **Add input videos**:  
   Place your input video files inside:

```
/data/raw/
```

4. **Set filenames in `main.py`**:  
   Inside `main.py`, update the input filenames to match the names of the videos you placed in `raw`.

5. **Run the script**:

```bash
python src/main.py
```

6. **Check the results**:  
   The output tracked videos will be saved in:

```
/data/processed/
```

---

##  Model Used

We use a YOLOv5 model file:

```
models/yolov5su.pt
```

Make sure this file is present in the correct folder when running the script.

---

## 🎥 Demo Videos

### 🔹 Input Video 1 Example  
[Click here to watch Input Video 1](https://drive.google.com/file/d/1TT4ljasEJdqE3umXK8vg1KWBEleMVYrj/view?usp=sharing)

### 🔹 Input Video 2 (Real Human Video)  
[Click here to watch Input Video 2](https://drive.google.com/file/d/11LcIgdHByzZRFkIC4NE1ai1hS1RaBIsL/view?usp=sharing)

---

### 🔹 Output Video 1 Example  
[Click here to watch Output Video 1](https://drive.google.com/file/d/1jKdwt5xX25VvC5G7UQjYQJmJ54lCsj1M/view?usp=sharing)

### 🔹 Output Video 2 Example  
[Click here to watch Output Video 2](https://drive.google.com/file/d/1enX1FkxwYi3zOO8gilejoo-VG6XUC5i4/view?usp=sharing)

---

## 📌 Notes

- Input 1 was a sample video from the internet.
- Input 2 was a real-world video recorded by me for this project.

