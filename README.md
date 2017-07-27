Optical Character Recognition in Chinese Manuscripts
======================================================

Description
------------
This project implements OCR (Optical Character Recognition) for chinese manuscripts. OCR in general is a popular research problem in image processing and pattern recognition. In particular, chinese letters make it even more complex by their high amount of variation in rotation, scale, line strength and several other features. The goal of this project was to determine whether two manuscripts were written by the same author. Solving this problem, the algorithm first performs several morphological transformation which then are folllowed by aligned intensity projections in order to obtain a reliable segmentation of letters. Afterwards, 10-dimensional feature vectors for all detected letters are computed and classified regarding their similarity by a slightly modified version of the k-Nearest-Neighbor algorithm modelling a normal distribution. Ultimately, the majority vote of detected matching letters among all gives rise to the confidence rate of two manuscripts being written by the same author. The project was implemented using Visual Studio with C++ and the OpenCV framework. It was part of the Master's course "Image Processing" at the University of Hamburg and has been awarded as the most robust, adaptive and accurate approach.

The pictures below illustrate the segmentation and detection process, followed by a recognition for writer-similarity identification.
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/Original.jpg" width="100%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/Morphology.jpg" width="100%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/Intensity-Threshold.jpg" width="100%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/kNN-Noise-Filter.jpg" width="100%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/Intensity-Projection.jpg" width="100%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/Detection.jpg" width="100%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/Steps/IP.jpg" width="100%">

Demo
------------
-Start WINDOWS (or anything that can execute .exe) :-)
-Open cmd and navigate to the OCR.exe folder
-Type: OCR.exe <imagepath1> <imagepath2> <k> <sigma> <output>

imagepath1 & imagepath2 -> paths to your .jpg manuscript
k -> parameter for kNN classification
sigma -> parameter for kNN decision-confidence threshold
output -> 0 or 1
	  0 if you only want to see the results
          1 if you also want to see the processing steps and probability distributions

Example:
OCR.exe front/3.jpg back/3.jpg 50 0.25 1
