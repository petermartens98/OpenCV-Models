# OpenCV Photo to Watercolor
# App Version 2

# Import necessary libraries
import cv2
import math

# Function that take photo file path and a zoom factor to adjust photo output size
def path_to_watercolor(in_path, zoom):
    # Define image output path
    out_path = in_path[:-4]+"_watercolor.jpg"

    # Read in image
    image = cv2.imread(in_path)

    # Scale and resize image for optimal size for processing
    scale = float(3000)/(image.shape[0]+image.shape[1])
    image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))

    # Create washout color effect
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    adjust_v = (image_hsv[:,:,2].astype("uint")+5)*3
    adjust_v = ((adjust_v > 255)*255 + (adjust_v <= 255)*adjust_v).astype("uint8")
    image_hsv[:,:,2]=adjust_v
    image_soft = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    image_soft = cv2.GaussianBlur(image_soft, (51,51),0)

    # Create outline sketch effect
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    invert = cv2.bitwise_not(image_gray)
    blur = cv2.GaussianBlur(invert, (21,21),0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(image_gray, inverted_blur, scale=265.0)
    sketch = cv2.merge([sketch,sketch,sketch])

    # Combine effects
    image_water = ((sketch/255.0)*image_soft).astype("uint8")

    # Save watercolor image to output file
    cv2.imwrite(out_path, image_water)

    # Function to rescale frame size
    def rescale_frame(frame, percent=100):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    image_water = rescale_frame(image_water, percent=zoom)

    # Display watercolor image
    cv2.imshow('Photo to Watercolor', image_water)
    cv2.waitKey()

# Define image input
in_path = "C:\\Users\\Peter\\PycharmProjects\\Photo_To_Watercolor\\PeterTrainingCamp.jpg"

# Run Function
path_to_watercolor(in_path, 75)

print("Code Completed")
