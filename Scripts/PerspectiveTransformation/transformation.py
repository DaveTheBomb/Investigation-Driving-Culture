import cv2
import numpy as np

def perspective_transform(image_path, coordinates):
    # Load the image
    image = cv2.imread(image_path)

    # Define the coordinates of the four points for perspective transformation
    pts_src = np.array(coordinates, dtype=np.float32)

    # Define the coordinates for the four corners of the desired output
    width = 850  # Adjust this value as needed
    height = 700  # Adjust this value as needed
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the perspective transformation
    result = cv2.warpPerspective(image, matrix, (width, height))


    # since bird eye view has property that all points are equidistant 
    # in horizontal and vertical direction. distance_v and distance_h will
    # give us 6ft distance in both horizontal and vertical directions
    # which we can use to calculate distance between two humans in bird eye view
    distance_v = np.sqrt((result[0][0] - result[1][0]) ** 2 + (result[0][1] - result[1][1]) ** 2)
    distance_h = np.sqrt((result[0][0] - result[2][0]) ** 2 + (result[0][1] - result[2][1]) ** 2)



    # Display the original and transformed images
    cv2.imwrite("Original Image", image)
    cv2.imwrite("Bird's Eye View", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'image_path' with the path to your input image
    image_path = 'output_image.jpeg'


    Points = [[325, 231], [447, 229] , [847, 581], [111, 575]]

    x1 = Points[0][0]
    y1 = Points[0][1]
    
    x2 = Points[1][0]
    y2 = Points[1][1]

    x3 = Points[2][0]
    y3 = Points[2][1]

    x4 = Points[3][0]
    y4 = Points[3][1]



    # Replace the coordinates with the four points of interest
    coordinates = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    perspective_transform(image_path, coordinates)