import cv2
import numpy as np
import pandas as pd
import math
from collections import Counter


#HELPER FUNCTION FOR SHAPE CONFIGURATION - MAJORITY BLACK PIXELS
def bp(matrix):
    black_pixels = np.sum(matrix == 0)  
    total_pixels = matrix.size         
    return black_pixels > (total_pixels / 2)  

#HELPER FUNCTION FOR CONTOUR SHAPE
def shape_category(contour, image):

    x, y, w, h = cv2.boundingRect(contour) # Find the bounding rectangle around the contour 
    rect = image[y:y+h, x:x+w] # Extract the part of the input image corrensponding to the rectangle
    cell_width = w // 4 # Calculate the width and height of each grid cell
    cell_height = h // 3

    # Create arrays to hold the pixels of each section
    sec1 = rect[0:cell_height, 0:cell_width]
    sec2 = rect[0:cell_height, cell_width:cell_width*2]
    sec3 = rect[0:cell_height, cell_width*2:cell_width*3]
    sec4 = rect[0:cell_height, cell_width*3:w]
    sec5 = rect[cell_height:cell_height*2, 0:cell_width]
    sec6 = rect[cell_height:cell_height*2, cell_width:cell_width*2]
    sec7 = rect[cell_height:cell_height*2, cell_width*2:cell_width*3]
    sec8 = rect[cell_height:cell_height*2, cell_width*3:w]
    sec9 = rect[cell_height*2:cell_height*3, 0:cell_width]
    sec10 = rect[cell_height*2:cell_height*3, cell_width:cell_width*2]
    sec11 = rect[cell_height*2:cell_height*3, cell_width*2:cell_width*3]
    sec12 = rect[cell_height*2:cell_height*3, cell_width*3:w]

    #determine the specs for each shape (more explanation in writeup)
    if (bp(sec5) and bp(sec6)):
        return "C-shaped"
    elif (bp(sec5) and bp(sec8)):
        return "I-shaped"
    elif (bp(sec7)):
        return "L-shaped"
    elif (bp(sec5) == False and bp(sec8) == False):
        return "rectangular"
    elif ( bp(sec1)!= bp(sec12) or bp(sec9)!= bp(sec4)):
        return "asymmetrical"

#HELPER FUNCTION FOR BOX_SIZE
def get_box_size(area):
    if area <= 2000:
        if area <=200:
            return "Smallest"
        else:
            return "Small"
    elif area > 2000 and area <= 5000:
        return "Medium"
    elif area > 5000:
        if area > 12000:
            return "Largest"
        else:
            return "Large"

#HELPER FUCNTION FOR BOX ASPECT RATIO
def aspect_ratio(box):
    width = abs(box[0][0] - box[1][0])
    height = abs(box[0][1] - box[1][1])
    diagonal = int(np.sqrt((width ** 2) + (height ** 2)))

    if height != 0:
        aspect_ratio = width / height
    else:
        aspect_ratio = 0
   
    if aspect_ratio < 0.75:
        return "narrow"
    elif aspect_ratio > 1.33:
        return "wide"
    else:
        return "medium-width"

#HELPER FUNCTION FOR BOX OVERLAP
def check_overlap(rect1, rect2):

    rect1_ur = rect1[0]
    rect1_ll = rect1[1]
    rect2_ur = rect2[0]
    rect2_ll = rect2[1]
    
    # Check for overlap
    if rect1_ur[0] < rect2_ll[0] or rect2_ur[0] < rect1_ll[0]:
        return False # Rectangles don't overlap horizontally
    if rect1_ur[1] < rect2_ll[1] or rect2_ur[1] < rect1_ll[1]:
        return False # Rectangles don't overlap vertically
    return True

#HELPER FUNCTION FOR VERTICAL LOCATION 
def get_vert_section(image, contour):
    
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Calculate the bounds of each section
    uppermost_bounds = (0, 0, width, height // 5)
    upper_bounds = (0, height // 5, width, height // 5)
    mid_height_bounds = (0, 2 * height // 5, width, height // 5)
    lower_bounds = (0, 3 * height // 5, width, height // 5)
    lowermost_bounds = (0, 4 * height // 5, width, height // 5)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the center of the bounding rectangle
    center_x = x + w // 2
    center_y = y + h // 2

    # Check which section the center of the bounding rectangle falls within
    if center_y >= uppermost_bounds[1] and center_y <= uppermost_bounds[1] + uppermost_bounds[3] and center_x >= uppermost_bounds[0] and center_x <= uppermost_bounds[0] + uppermost_bounds[2]:
        return 'uppermost'
    elif center_y >= upper_bounds[1] and center_y <= upper_bounds[1] + upper_bounds[3] and center_x >= upper_bounds[0] and center_x <= upper_bounds[0] + upper_bounds[2]:
        return 'upper'
    elif center_y >= mid_height_bounds[1] and center_y <= mid_height_bounds[1] + mid_height_bounds[3] and center_x >= mid_height_bounds[0] and center_x <= mid_height_bounds[0] + mid_height_bounds[2]:
        return 'mid-height'
    elif center_y >= lower_bounds[1] and center_y <= lower_bounds[1] + lower_bounds[3] and center_x >= lower_bounds[0] and center_x <= lower_bounds[0] + lower_bounds[2]:
        return 'lower'
    elif center_y >= lowermost_bounds[1] and center_y <= lowermost_bounds[1] + lowermost_bounds[3] and center_x >= lowermost_bounds[0] and center_x <= lowermost_bounds[0] + lowermost_bounds[2]:
        return 'lowermost'
    else:
        return 'unknown'

#HELPER FUNCTION FOR HORIZONTAL LOCATION 
def get_hoz_section(image, contour):
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Calculate the bounds of each section
    leftmost_bounds = (0, 0, width // 5, height)
    left_bounds = (width // 5, 0, width // 5, height)
    mid_width_bounds = (2 * width // 5, 0, width // 5, height)
    right_bounds = (3 * width // 5, 0, width // 5, height)
    rightmost_bounds = (4 * width // 5, 0, width // 5, height)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the center of the bounding rectangle
    center_x = x + w // 2
    center_y = y + h // 2

    # Check which section the center of the bounding rectangle falls within
    if center_x >= leftmost_bounds[0] and center_x <= leftmost_bounds[0] + leftmost_bounds[2] and center_y >= leftmost_bounds[1] and center_y <= leftmost_bounds[1] + leftmost_bounds[3]:
        return 'leftmost'
    elif center_x >= left_bounds[0] and center_x <= left_bounds[0] + left_bounds[2] and center_y >= left_bounds[1] and center_y <= left_bounds[1] + left_bounds[3]:
        return 'left'
    elif center_x >= mid_width_bounds[0] and center_x <= mid_width_bounds[0] + mid_width_bounds[2] and center_y >= mid_width_bounds[1] and center_y <= mid_width_bounds[1] + mid_width_bounds[3]:
        return 'mid-width'
    elif center_x >= right_bounds[0] and center_x <= right_bounds[0] + right_bounds[2] and center_y >= right_bounds[1] and center_y <= right_bounds[1] + right_bounds[3]:
        return 'right'
    elif center_x >= rightmost_bounds[0] and center_x <= rightmost_bounds[0] + rightmost_bounds[2] and center_y >= rightmost_bounds[1] and center_y <= rightmost_bounds[1] + rightmost_bounds[3]:
        return 'rightmost'
    else:
        return 'unknown'

#HELPER FUNCTION FOR ORIENTATION
def get_orientation(contour, threshold=0.1):

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the aspect ratio of the bounding rectangle
    aspect_ratio = w / h

    # Check if the aspect ratio is within the threshold of 1.0 (i.e., non-oriented)
    if abs(aspect_ratio - 1.0) < threshold:
        return 'non-oriented'
    # Check if the aspect ratio is greater than 1.0 (i.e., horizontally-oriented)
    elif aspect_ratio > 1.0:
        return 'horizontally-oriented'
    # Otherwise, the aspect ratio is less than 1.0 (i.e., vertically-oriented)
    else:
        return 'vertically-oriented'

#FUNCTION TO CREATE "CONFUSION" ARRAYS FOR A GIVEN METRIC
def get_twins(metric):
    for obj_name in grand_dict:
        twins = []
        for other_name in grand_dict:
            if other_name != obj_name: 
                if (grand_dict[obj_name][metric] == grand_dict[other_name][metric]):
                    twins.append(other_name)  
        grand_dict[obj_name].append(twins)

#FUNCTION TO COMPUTE NEARNESS FOR TWO CONTOURS
def calculate_nearness(contour1, contour2):

    # Create bounding rectangles for the contours
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    
    # Calculate the points along the bounding rectangles
    rect1_pts = [(x1, y1), (x1+w1, y1), (x1, y1+h1), (x1+w1, y1+h1)]
    rect2_pts = [(x2, y2), (x2+w2, y2), (x2, y2+h2), (x2+w2, y2+h2)]
    
    # Calculate the minimum distance between the two bounding rectangles
    min_dist = float('inf')
    for pt1 in rect1_pts:
        for pt2 in rect2_pts:
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist < min_dist:
                min_dist = dist
    
    # Normalize the distance between 0 and 1 based on the maximum possible distance
    max_dist = np.linalg.norm(np.array([w1+h1, w2+h2]))
    norm_dist = min_dist / max_dist
    
    return norm_dist

#FINAL OUTPUT FORMATTING FUNCTION
def building_near(building, arr1, arr2, arr3, arr4, arr5):
    # Create nearTo array
    nearTo = []
    if arr3:
        nearTo = arr3[:2]
    elif arr4:
        nearTo = arr4[:2]
    elif arr5:
        nearTo = arr5[:2]

    # Create output string
    output = f"{building}: {arr1[0]}, {arr1[1]}, {arr1[2]}, {arr2[0]}, {arr2[1]}, {arr2[2]} building near to "
    if nearTo:
        output += f"{nearTo[0]}"
    if len(nearTo) > 1:
        output += f" and {nearTo[1]}"
    return output

#################################################################################################
#################################################################################################

# INITIALIZE DICTIONARY
grand_dict = {}

# READ IN TABLE.TXT TO POPULATE NAMELIST 
namelist = [] 
with open("Table.txt") as f:
    lines = f.readlines()
for line in lines:
    items = line.split() # Split the line by whitespace
    name = items[1] # Store the second item as a string labeled "name"
    namelist.append(name)
namelist = namelist[::-1] # Reverse namelist to match contour read-in order in the next block
i = 0

# READ IMAGE + LOCATE SHAPES 
image = cv2.imread('Labeled.pgm', cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_height = image.shape[0]

# ADD SHAPES TO DICTIONARY 
for i, contour in enumerate(contours):

    # FIND OBJECT
    x, y, w, h = cv2.boundingRect(contour)
    obj = image[y:y+h, x:x+w]

    # CALCULATE OBJECT INTENSITY
    intensity = np.max(obj) 

    #ADD TO GRAND_DICTIONARY
    obj_name = namelist[i]  
    i += 1
    grand_dict[obj_name] = [intensity]

    #ADD CENTER OF MASS TO GRAND DICT VALUE
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    com = [cx, cy]
    grand_dict[obj_name].append(com)

    #ADD TOTAL PIX AREA TO GRAND DICT VALUE
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    area = int(rect[1][0] * rect[1][1])
    grand_dict[obj_name].append(area)

    #IMAGE TESTER LINE - GENERATE PIC WITH MBR BOXES 
    # cv2.drawContours(image, [box], 0, (255, 0, 0), 2) 

    #ADD L/R CORNERS + DIAG OF BOX TO GRAND DICT
    x_coords = [box[i][0] for i in range(4)]
    y_coords = [box[i][1] for i in range(4)]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    diag = int(np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)) 
    lowLeft = (x_min, y_min)
    upRight = (x_max, y_max) 
    corners = [upRight, lowLeft]
    grand_dict[obj_name].append(corners)
    grand_dict[obj_name].append(diag)

    #CLASSIFY BY SIZE, ASPECT RATIO, AND SHAPE
    what = [] 
    size = get_box_size(grand_dict[obj_name][2])
    what.append(size)
    aspect = aspect_ratio(grand_dict[obj_name][3])
    what.append(aspect)
    shape = shape_category(contour, image)
    what.append(shape)
    grand_dict[obj_name].append(what)

    #CLASSIFY BY LOCATION AND ORIENTATION
    where = []
    vert = get_vert_section(image, contour)
    where.append(vert)
    hoz = get_hoz_section(image, contour)
    where.append(hoz)
    ornt = get_orientation(contour)
    where.append(ornt)
    grand_dict[obj_name].append(where)

    grand_dict[obj_name].append(contour)

# CREATE NEIGHBORS ARRAY FOR EACH KEY IN GRAND_DICT
for obj_name in grand_dict:
    neighbors = []
    for other_name in grand_dict:
        if other_name != obj_name: 
            if check_overlap(grand_dict[other_name][3], grand_dict[obj_name][3]):
                neighbors.append(other_name)  
    grand_dict[obj_name].append(neighbors)

get_twins(5) # adds geometric twins to the end of the value array 
get_twins(6) # adds location twins to the end of the value array 

#  CREATE NEARNESS ARRAY FOR EACH KEY IN GRAND_DICT
for obj_name in grand_dict:
    nearness = []
    for other_name in grand_dict:
        if other_name != obj_name: 
            near = calculate_nearness(grand_dict[other_name][7], grand_dict[obj_name][7]) 
            if obj_name == "AlmaMater":
                if near < 0.3:
                    nearness.append(other_name) 
            if near < 0.165:
                nearness.append(other_name)  
    grand_dict[obj_name].append(nearness)
    
with open("output.html", "w") as f:
    for obj_name in grand_dict:
        arr1 = grand_dict[obj_name][5]     
        arr2 = grand_dict[obj_name][6]    
        arr3 = grand_dict[obj_name][8]    
        arr4 = grand_dict[obj_name][10]     
        arr5 = grand_dict[obj_name][11]    
        building = obj_name
        output = building_near(building, arr1, arr2, arr3, arr4, arr5)
        output = output.replace(building, f"<b>{building}</b>")
        f.write(f"{output}<br>")

#################################################################################################
#################################################################################################

# STEP 1 DELIVERABLES GENERATOR
data_list = []
for key in grand_dict:
    data_dict = {
        "intensity": grand_dict[key][0],
        "name": key,
        "(x,y) of CoM": grand_dict[key][1],
        "pixel area": grand_dict[key][2],
        "L/R corners": grand_dict[key][3],
        "MBR Diag": grand_dict[key][4],
        "Intersectors": grand_dict[key][8]
    }
    data_list.append(data_dict)
df = pd.DataFrame(data_list)
html = df.to_html(index=False, bold_rows=True)
with open("step1.html", "w") as f:
    f.write(html)

# STEP 2 DELIVERABLES GENERATOR 
data_list = []
for key in grand_dict:
    data_dict = {
        "name": key,
        "size": grand_dict[key][5][0],
        "aspect ratio": grand_dict[key][5][1],
        "shape": grand_dict[key][5][2],
        "confusion": grand_dict[key][9]
    }
    data_list.append(data_dict)
df = pd.DataFrame(data_list)
html = df.to_html(index=False, bold_rows=True)
with open("step2.html", "w") as f:
    f.write(html)

# STEP 3 DELIVERABLES GENERATOR 
data_list = []
for key in grand_dict:
    data_dict = {
        "name": key,
        "verticality": grand_dict[key][6][0],
        "horizontality": grand_dict[key][6][1],
        "orientation": grand_dict[key][6][2],
        "confusion": grand_dict[key][10]
    }
    data_list.append(data_dict)
df = pd.DataFrame(data_list)
html = df.to_html(index=False, bold_rows=True)
with open("step3.html", "w") as f:
    f.write(html)

# STEP 4 DELIVERABLES GENERATOR 
data_list = []
for key in grand_dict:
    data_dict = {
        "name": key,
        "near to": grand_dict[key][11],
    }
    data_list.append(data_dict)
df = pd.DataFrame(data_list)
html = df.to_html(index=False, bold_rows=True)
with open("step4.html", "w") as f:
    f.write(html)
