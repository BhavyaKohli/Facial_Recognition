# Facial_Recognition

This repository contains: <br>
\> the opencv face detection haarcascade_frontalface_alt.xml file, <br>
\> the test and train csv files, and <br>
\> the master.py python file, which will be used for performing the facial recognition. <br>

The modules required to run the file are: <br>
\> face_recognition, tqdm, sklearn*, sys <br>
\> numpy, pandas, matplotlib*, cv2 <br>
_* only for visualization and when true labels are given_

The file can be run using the following format: <br>
```python master.py (source csv file path) (source images path) (number of samples, if needed) (0 or blank for testing)```

so, for example, <br>
```python master.py ./train.csv ./dataset_images 2000 1```
