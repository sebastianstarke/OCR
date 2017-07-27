Optical Character Recognition in Chinese Manuscripts
======================================================
<img src ="https://github.com/sebastianstarke/OCR/blob/master/images/Data/back/4.jpg" width="45%">
<img src ="https://github.com/sebastianstarke/OCR/blob/master/images/Data/front/1.jpg" width="45%">

Description
------------
This project implements OCR (Optical Character Recognition) for chinese manuscripts. OCR in general is a popular research problem in image processing and pattern recognition. In particular, chinese letters make it even more complex by their high amount of variation in rotation, scale, line strength and several other features. The goal of this project was to determine whether two manuscripts were written by the same author. Solving this problem, the algorithm first performs several morphological transformation which then are folllowed by aligned intensity projections in order to obtain a reliable segmentation of letters. Afterwards, 10-dimensional feature vectors for all detected letters are computed and classified regarding their similarity by a slightly modified version of the k-Nearest-Neighbor algorithm modelling a normal distribution. Ultimately, the majority vote of detected matching letters among all gives rise to the confidence rate of two manuscripts being written by the same author. The project was implemented using Visual Studio with C++ and the OpenCV framework. It was part of the Master's course "Image Processing" at the University of Hamburg and has been awarded as the most robust, adaptive and accurate approach.

The pict
<img src ="https://github.com/sebastianstarke/BioIK/blob/master/images/1.png" width="100%">

