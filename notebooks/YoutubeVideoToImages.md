<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/notebooks/YoutubeVideoToImages.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
!ls
```

    drive  sample_data
    


```python
!ls 'drive/MyDrive/Projects/Video to Images'
```

    'StatQuest_ Linear Discriminant Analysis (LDA) clearly explained..mp4'
    


```python
folderpath = 'drive/MyDrive/Projects/Video to Images/'
filename = 'StatQuest_ Linear Discriminant Analysis (LDA) clearly explained..mp4'
```


```python
import cv2
import numpy as np

```


```python
import os
```


```python
# Load the video
video_path = os.path.join(folderpath, filename)
cap = cv2.VideoCapture(video_path)
```


```python
# Initialize variables
prev_frame = None
frame_count = 0
screenshot_count = 0
threshold = 30  # Sensitivity to detect changes, adjust as necessary

def save_screenshot(frame, count):
    screenshot_path = os.path.join(folderpath, f'screenshot_{count}.png')
    cv2.imwrite(screenshot_path, frame)
    print(f'Saved screenshot: {screenshot_path}')

# Read and process the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Compute the absolute difference between the current frame and previous frame
        diff = cv2.absdiff(prev_frame, gray_frame)
        # Threshold the difference
        _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        # Compute the percentage of changed pixels
        change_percentage = np.sum(diff_thresh) / (diff_thresh.shape[0] * diff_thresh.shape[1])

        if change_percentage > 0.01:  # Adjust this threshold as needed
            save_screenshot(frame, screenshot_count)
            screenshot_count += 1

    prev_frame = gray_frame

cap.release()
cv2.destroyAllWindows()
```

    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_0.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_1.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_2.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_3.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_4.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_5.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_6.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_7.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_8.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_9.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_10.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_11.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_12.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_13.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_14.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_15.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_16.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_17.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_18.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_19.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_20.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_21.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_22.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_23.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_24.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_25.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_26.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_27.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_28.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_29.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_30.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_31.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_32.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_33.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_34.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_35.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_36.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_37.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_38.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_39.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_40.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_41.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_42.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_43.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_44.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_45.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_46.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_47.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_48.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_49.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_50.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_51.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_52.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_53.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_54.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_55.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_56.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_57.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_58.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_59.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_60.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_61.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_62.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_63.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_64.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_65.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_66.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_67.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_68.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_69.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_70.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_71.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_72.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_73.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_74.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_75.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_76.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_77.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_78.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_79.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_80.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_81.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_82.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_83.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_84.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_85.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_86.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_87.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_88.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_89.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_90.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_91.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_92.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_93.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_94.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_95.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_96.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_97.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_98.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_99.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_100.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_101.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_102.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_103.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_104.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_105.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_106.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_107.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_108.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_109.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_110.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_111.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_112.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_113.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_114.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_115.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_116.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_117.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_118.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_119.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_120.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_121.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_122.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_123.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_124.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_125.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_126.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_127.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_128.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_129.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_130.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_131.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_132.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_133.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_134.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_135.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_136.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_137.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_138.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_139.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_140.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_141.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_142.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_143.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_144.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_145.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_146.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_147.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_148.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_149.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_150.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_151.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_152.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_153.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_154.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_155.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_156.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_157.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_158.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_159.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_160.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_161.png
    Saved screenshot: drive/MyDrive/Projects/Video to Images/screenshot_162.png
    
