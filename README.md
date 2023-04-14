# Visual-Interfaces-Proj3
This assignment explores how well an algorithm can describe an image in human language.

The aim of this project was to process an image of Columbia's campus map and extract information regarding each building's location. The script generates an English sentence for each building contour describing its position on campus. For instance, the output for the "Pupin" building was "the small, wide, rectangular, uppermost, left, horizontally-oriented building near Uris and the Northwest Corner Building."

Each building was extracted from the image as an OpenCV contour, then shape descriptors were generated using a combination of contour area, height, width, and a method where I created a bounding rectangle around each contour, divided it into four columns and three rows, then based on which cells the contour’s pixels filled, the shape would be classified as “rectangular, C-shaped, L-shaped, or I-shaped”. 

The location metrics were computed by dividing the campus map into five horizontal and vertical sections. The absolute location of each building was then determined based on which section its center of mass fell into.

To determine the proximity of each building relative to the others, the code employs three methods. The first method involves analyzing similarities between each building’s location metrics mentioned above. The second checks for overlapping bounding rectangles between contours. Finally, the third algorithm calculates the distance between every contour and every other contour, returning information on the relative closeness of each building.

For a more detailed explanation of the algorithms used to generate these metrics, feel free to refer to the included writeup.

