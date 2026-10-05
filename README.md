# Smart Attendance System using OpenCV

An automated attendance system that marks attendance using face detection and recognition. Built as MCA Mini  Project.

**Features**
- Real-time face detection using Haar Cascade Classifier
- Face recognition and identification of students
- Automatic attendance marking with date & time
- CSV/Excel attendance report generation
- Eliminates proxy attendance

**Tech Stack**
- Python
- OpenCV (cv2)
- Haar Cascade for face detection
- NumPy, Pandas
- CSV for data storage

**How It Works**
1.  Detects face from webcam using Haar Cascade
2.  Recognizes face from trained dataset
3.  Marks attendance automatically in sheet
4.  Generates daily report

**How to Run**
```bash
pip install opencv-python numpy pandas
python attendance.py
