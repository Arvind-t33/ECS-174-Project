import cv2
import numpy as np
import pytesseract

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def detect_text(image):
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=custom_config)
    return details

def main():
    # Read the image
    image = cv2.imread('chinese_signs/images/Train_CD_026.jpg')

    # Define the lower and upper bounds for blue color in HSV
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for blue color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask_blue)

    # Convert the masked image to grayscale
    gray = preprocess_image(masked_image)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobel_x, sobel_y)

    # Normalize edges
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Iterate through the filtered contours
    for contour in filtered_contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio of the bounding box
        aspect_ratio = float(w) / h

        # Calculate the area of the contour and the bounding box
        contour_area = cv2.contourArea(contour)
        bounding_box_area = w * h

        # Calculate extent
        extent = float(contour_area) / bounding_box_area

        # Check if the aspect ratio and extent are within looser acceptable ranges
        if 0.5 <= aspect_ratio <= 2.0 and extent > 0.5:
            # Draw the bounding box on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Preprocess the cropped image
            roi = image[y:y+h, x:x+w]
            preprocessed_roi = preprocess_image(roi)

            # Detect text in the cropped image
            details = detect_text(preprocessed_roi)
            
            # Draw text bounding boxes and print text
            n_boxes = len(details['level'])
            for i in range(n_boxes):
                if details['text'][i].strip():
                    (x_text, y_text, w_text, h_text) = (details['left'][i] + x, details['top'][i] + y, 
                                                        details['width'][i], details['height'][i])
                    image = cv2.rectangle(image, (x_text, y_text), (x_text + w_text, y_text + h_text), (0, 255, 0), 2)
                    print(details['text'][i])

    # Display the result
    cv2.imshow('Rectangles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



