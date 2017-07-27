====================HOW-TO-USE====================
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
