import cv2
import numpy as np

# Function defined to find the centre of any contour then return its x and y coordiantes as a tuple
def findCentre(contour):
    M = cv2.moments(contour)
    area = M['m00']
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)
###################### PART I: TEETH INSPECTION ###########################################
############ READ THE IDEAL AND SAMPLE IMAGES (GRAYSCALE) AND CREATE MASK #################
sample_image = "sample6.jpg"
ideal_image = cv2.imread("sample1.jpg")
ideal = cv2.cvtColor(ideal_image, cv2.COLOR_BGR2GRAY)
sample = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)

# Using a threshold of 30 removes most of the noise when creating the binary mask
ret, ideal_threshold = cv2.threshold(ideal, 30, 255, cv2.THRESH_BINARY)
ret, sample_threshold = cv2.threshold(sample, 30, 255, cv2.THRESH_BINARY)


############ GET THE BINARY IMAGE OF THE IDEAL TEETH #################
ideal_teeth = cv2.cvtColor(ideal_image, cv2.COLOR_BGR2GRAY)

# Draw 2 black circles to remove everything except the teeth
cx, cy = findCentre(ideal_threshold)
cv2.circle(ideal_teeth,(cx - 6,cy),165,(0, 0, 0),-1)
cv2.circle(ideal_teeth,(cx + 6,cy),165,(0, 0, 0),-1)

# Form the binary image
ret, ideal_teeth_threshold = cv2.threshold(ideal_teeth, 30, 255, cv2.THRESH_BINARY)

# Erode to remove noise and the connection of the teeth. Done to find contours for each individual tooth
ideal_teeth_erosion = cv2.erode(ideal_teeth_threshold, np.ones((3,3), np.uint8), iterations = 1)
contours_ideal_teeth, _ = cv2.findContours(ideal_teeth_erosion,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


############ GET THE BINARY IMAGE OF THE FAULTY TEETH #################
# Find the difference between the ideal mask and the sample mask to get faulty part of teeth
difference = cv2.bitwise_xor(ideal_threshold, sample_threshold)

# Draw 2 black circles to remove everything except the teeth
cv2.circle(difference,(cx - 6,cy),165,(0, 0, 0),-1)
cv2.circle(difference,(cx + 6,cy),165,(0, 0, 0),-1)

# Erode to remove noise and the connection of the teeth. Done to find contours for each individual tooth
difference_erosion = cv2.erode(difference, np.ones((3,3), np.uint8), iterations = 1)
contours_difference, _ = cv2.findContours(difference_erosion,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


############ DRAW A BOUNDING CIRCLE FOR EACH FAULTY TOOTH #################
contours_difference_centres = []
for c in contours_difference:
    cx,cy =findCentre(c)
    cv2.circle(difference_erosion, (cx,cy),22,(255,255,255),-1)
    # Save the centres of each faulty tooth in a list to be matched with the ideal teeth
    contours_difference_centres.append((cx, cy))


########### USE BOUNDING CIRCLE TO MATCH IDEAL TEETH WITH BROKEN TEETH ####################
ideal_teeth_filtered = cv2.bitwise_and(difference_erosion, ideal_teeth_erosion)
contours_ideal_teeth_filtered , _ = cv2.findContours(ideal_teeth_filtered ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


########### FIND THE CENTRES OF THE FILTERED IDEAL TEETH FOR COMPARISON ##################
contours_ideal_teeth_filtered_centres = []
for c in contours_ideal_teeth_filtered:
    cx,cy =findCentre(c)
    contours_ideal_teeth_filtered_centres.append((cx,cy))


############# COMPARE CENTRES TO MATCH THE IDEAL TEETH TO FAULTY TEETH ################
contours_ideal_teeth_matched = []
count = 0

# Comparing the centre of a faulty teeth to all the available ideal teeth by
# calculating the centre distance between the faulty tooth and each of the ideal teeth
for i in contours_difference_centres:
    diff_x = i[0]
    diff_y = i[1]
    for j in contours_ideal_teeth_filtered_centres:
        ideal_x = j[0]
        ideal_y = j[1]
        centre_distance = ((ideal_x-diff_x)**2 + (ideal_y - diff_y)**2) ** 0.5
        if centre_distance < 20:
            contours_ideal_teeth_matched.append(contours_ideal_teeth_filtered[count])
            break # speeds up the process, since only a single centre distance satisfies condition
        count += 1
    count = 0 # count variable is used as the index for the contours list


################ WORN OUT (blue) OR BROKEN (red) #####################
# Define the variables to count the faulty teeth into two categories
worn_out_teeth = 0
broken_teeth = 0
teeth_image = ideal_image

# Compare the area of the faulty tooth to the matching ideal tooth
# if the area of each is almost the same, the tooth is likely to be broken
# if the area of each is significantly different, the tooth is likey to be worn out
for i in range (0,len(contours_difference)):
    area_difference = cv2.contourArea(contours_difference[i])
    area_ideal = cv2.contourArea(contours_ideal_teeth_matched[i])
    if(area_difference / area_ideal < 0.85):
        worn_out_teeth  += 1
        cx,cy = findCentre(contours_difference[i])
        cv2.circle(teeth_image, (cx,cy),10,(255,0,0),3)
    else:
        broken_teeth +=1
        cx, cy = findCentre(contours_difference[i])
        cv2.circle(teeth_image, (cx, cy), 15, (0, 0, 255), 5)


########################## PART II: INNER DIAMETER ###########################################
# Reread the images to remove previous edits
ideal_image = cv2.imread("sample1.jpg")
sample_image = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)
ideal_diameter = cv2.cvtColor(ideal_image, cv2.COLOR_BGR2GRAY)

# Recreate the binary masks, this time with inverse thresholding
# This is done so that the inner circle will be white and so, its contours can be found
ret, ideal_diameter_threshold = cv2.threshold(ideal_diameter, 30, 255, cv2.THRESH_BINARY_INV)
ret, sample_threshold = cv2.threshold(sample_image, 30, 255, cv2.THRESH_BINARY_INV)

# Draw 1 black circle with high thickness to remove everything except the centre part of gear
cx, cy = findCentre(ideal_threshold)
cv2.circle(ideal_diameter_threshold, (cx,cy), 300, (0,0,0), 500) # removes white background
cv2.circle(sample_threshold, (cx,cy), 300, (0,0,0), 500)

# Find the contours of the inner circle of the ideal gear and the sample gear
contours_ideal_diameter, _ = cv2.findContours(ideal_diameter_threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_sample, _ = cv2.findContours(sample_threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Using contourArea, find the area of the inner circle of ideal gear, which will be used
# to find the the radius in image corresponding to 30 mm
# print(cv2.contourArea(contours_ideal_diameter[0])) # area of ideal circle = 1963.5 equals to 25 radius circle
# diameter = 50 units 52.5 47.5
# 50 + 5% = 52.5 / 2 = 26.25
# 50 - 5% = 47.5 / 2 = 23.75


######################### DRAW LIMIT CIRCLES ###############################
cx,cy = findCentre(contours_ideal_diameter[0])
cv2.circle(ideal_image, (cx, cy), 27, (0, 0, 255), 1) # rounded up for clarity and only able to input int radius
cv2.circle(ideal_image, (cx, cy), 23, (0, 0, 255), 1)  # rounded down for clarity and only able to input int radius


######################## CHECK VALIDITY OF INNER DIAMETER ##################
if len(contours_sample) > 0: # check done if there is no inner opening to prevent errors
    cv2.drawContours(ideal_image, contours_sample[0], -1, (0, 255, 0), 2)
    # Compare area to upper and lower limit circles
    if 3.14 * 23.75**2 < cv2.contourArea(contours_sample[0]) < 3.14 * 26.25**2:
        print("Inner diameter within allowable range!")
    else:
        print("Inner diameter is faulty and not within allowable range!")
else:
    print("Inner diameter is faulty and not within allowable range!")


###################### OUTER GEAR DIAMETER #################################
# Outer diameter of circle, 195 was found through trial and error and was chosen based
# on passing the tip of most teeth
# 195 units translates to 195 units * 15 mm / 25 units ~= 117 mm * 2 = 234 mm outer diameter
cv2.circle(ideal_image, (cx, cy), 195, (255, 0, 0), 1)
print("Outer gear diamater: "  + str(195 * 30 /25 ))


###################### FINAL OUTPUT ########################################
print("Broken Teeth: " + str(broken_teeth))
print("Worn-out Teeth: " + str(worn_out_teeth))
cv2.imshow("Defect Localization", teeth_image)
cv2.imshow("Diameter Verification", ideal_image)
cv2.waitKey(0)

