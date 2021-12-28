# Human-Detection-using-HOG

Instructions to run the program:
1. Download OpenCV for python, which can be directly added on Anaconda Navigator or
by executing the following command in the terminal: pip install opencv-python.
(Note: OpenCV package is just used to read and write images in the source code)
2. The other packages that must be installed are Math, Numpy and Os, Glob.
3. Open the source code files in a code editor which supports python. They are HOG.py,
KNN.py, GradMag.py and ascii.py.


To run the detection on test image:
4. Open HOG.py and compile (All the .py files must be inside the same folder).
5. In the KNN.py file, paste the path for positive and negative training images on line 20
and 21 respectively.
6. Similarly, paste the test image location path on line 16 and 17.
7. Run the Program to check the detection results in the output terminal.


To check the gradient magnitude images.
8. Open GradMag.py and paste the image path for positive and negative test dataset on line
14 and 15 respectively.
9. Run the program.
10. The Gradient Images will be stored in the folder containing the positive and negative test
images.


To get the HOG Vectors in .txt files
11. Open the ascii.py file, paste the path for positive and negative training images on line 16
and 17 respectively.
12. Similarly, paste the test image location path on line 12 and 13.
13. Run the program.
14. The .txt files will be saved in the folder containing all the python scripts.
