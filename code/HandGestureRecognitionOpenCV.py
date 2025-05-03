import cv2
import numpy as np
import math

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, img = cap.read()
    if not ret or img is None:
        print("Failed to capture frame")
        continue

    # Draw a larger Region of Interest (ROI) rectangle on the frame
    cv2.rectangle(img, (450, 450), (30, 30), (0, 255, 0), 2)
    # Crop the ROI from the image
    crop_img = img[30:450, 30:450]

    # Convert cropped image to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grey, (35, 35), 0)
    # Apply Otsu's thresholding (inverted binary)
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    # Find contours from the thresholded image
    contours = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    if len(contours) == 0:
        # If no contours found, just show the default windows and continue
        cv2.imshow('Gesture', img)
        cv2.imshow('Contours', crop_img)
        if cv2.waitKey(1) == 27:  # Exit if ESC is pressed
            break
        continue

    # Get the largest contour (presumably the hand)
    cnt = max(contours, key=cv2.contourArea)

    # Draw bounding box around the hand
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw the contour and its convex hull
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)

    # Find convexity defects (points between fingers)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0

    # Draw all contours on the thresholded image (for reference)
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    if defects is not None:
        # Loop through all defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # Calculate the sides of the triangle using Euclidean distance
            a = math.dist(end, start)
            b = math.dist(far, start)
            c = math.dist(end, far)

            # Use cosine rule to find the angle
            if b != 0 and c != 0:
                angle = math.acos((b**2 + c**2 - a**2)/(2 * b * c)) * 57

                # Count as a defect if angle < 90 degrees
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_img, far, 4, [0, 0, 255], -1)

            # Draw line between start and end points
            cv2.line(crop_img, start, end, [0, 255, 0], 2)

    # Set text properties for displaying gesture count
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (0, 255, 255)
    position = (60, 60)

    # Display text based on number of defects (fingers)
    if count_defects == 1:
        cv2.putText(img, "GESTURE ONE", position, font, font_scale, color, thickness, cv2.LINE_AA)
    elif count_defects == 2:
        cv2.putText(img, "GESTURE TWO", position, font, font_scale, color, thickness, cv2.LINE_AA)
    elif count_defects == 3:
        cv2.putText(img, "GESTURE THREE", position, font, font_scale, color, thickness, cv2.LINE_AA)
    elif count_defects == 4:
        cv2.putText(img, "GESTURE FOUR", position, font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(img, "Hello World!!!", position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Show the final output windows
    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    # Break the loop when ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
