import cv2
import pytesseract

# Read the image file
image = cv2.imread(r"C:\Users\acer\Desktop\licensePlateReader\2.JPG")
# image = imutils.resize(image, width=760)
# Convert to Grayscale Image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('',gray_image)



#Canny Edge Detection
canny_edge = cv2.Canny(gray_image, 170, 200)
cv2.imshow('',canny_edge)


# Find contours based on Edges
contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30]
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.imshow('',image)
print(len(contours))
# Initialize license Plate contour and x,y coordinates
contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

# Find the contour with 4 potential corners and creat ROI around it
for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        # print(len(approx))
        if len(approx) == 4: #see whether it is a Rectangle
            # cv2.drawContours(image,[approx],-1,(0,255,0), 2)
            # cv2.imshow('',image)
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2) 
            license_plate = gray_image[y:y + h, x:x + w]
            break
print("contour=" +str(len(contours)))
# print(contours[0])
# cv2.drawContours(image,[approx],-1,(0,255,0), 3)
# cv2.imshow('',image)

# Removing Noise from the detected image, before sending to Tesseract
license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
(thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)
# cv2.imshow('',license_plate)

#Text Recognition
text = pytesseract.image_to_string(license_plate)
#Draw License Plate and write the Text
image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 3) 
image = cv2.putText(image, '', (x-100,y-50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 6, cv2.LINE_AA)

print("License Plate :", text)


cv2.imshow("License Plate Detection",image)
cv2.waitKey(0)