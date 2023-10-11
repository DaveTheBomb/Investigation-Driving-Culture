import cv2
import numpy
from rembg import remove

def remove_background(input_path, output_path):
    # Read the input image using OpenCV
    image = cv2.imread(input_path)

    # Use the rembg library to remove the background
    with open(input_path, 'rb') as inp_file:
        input_data = inp_file.read()
        output_data = remove(input_data)

    # Save the resulting image with the background removed
    with open(output_path, 'wb') as out_file:
        out_file.write(output_data)

    # Read the resulting image with the background removed
    result_image = cv2.imdecode(
        numpy.frombuffer(output_data, numpy.uint8), cv2.IMREAD_UNCHANGED
    )

    # Display the original and processed images using OpenCV
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', image)

    cv2.namedWindow('Image with Background Removed', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Background Removed', result_image)

    # Handle keyboard events and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_image_path = 'sample_frame.jpeg'
    output_image_path = 'output_image.jpeg'

    remove_background(input_image_path, output_image_path)