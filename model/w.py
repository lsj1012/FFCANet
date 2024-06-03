# import cv2
#
#
# def draw_grid(image, rows, cols, color=(0, 255, 0), thickness=1):
#     """Draws a grid on the given image."""
#     height, width, _ = image.shape
#     row_step = height // rows
#     col_step = width // cols
#
#     # Draw horizontal lines
#     for i in range(1, rows):
#         cv2.line(image, (0, i * row_step), (width, i * row_step), color, thickness)
#
#     # Draw vertical lines
#     for j in range(1, cols):
#         cv2.line(image, (j * col_step, 0), (j * col_step, height), color, thickness)
#
#     return image
#
#
# # Load an image
# image_path = 'E:/Ultra-Fast-Lane-Detection-master/1.jpg'  # Replace 'your_image_path.jpg' with the path to your image
# image = cv2.imread(image_path)
#
# # Check if image loaded successfully
# if image is None:
#     print("Error: Unable to load image.")
# else:
#     # Specify the number of rows and columns for the grid
#     rows = 30
#     cols = 20
#
#     # Draw the grid
#     image_with_grid = draw_grid(image.copy(), rows, cols)
#
#     # Display the image with grid
#     cv2.imshow('Image with Grid', image_with_grid)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2


def draw_grid(image, rows, cols, color=(0, 255, 0), thickness=1):
    """Draws a grid on the given image."""
    height, width, _ = image.shape
    row_step = height // rows
    col_step = width // cols

    # Calculate the starting row position for the grid
    start_row = height // 4  # 3/4 height

    # Draw horizontal lines
    for i in range(1, rows):
        cv2.line(image, (0, start_row + i * row_step), (width, start_row + i * row_step), color, thickness)

    # Draw vertical lines
    for j in range(1, cols):
        cv2.line(image, (j * col_step, start_row), (j * col_step, start_row + height), color, thickness)

    return image


# Load an image
image_path = 'E:/Ultra-Fast-Lane-Detection-master/1.jpg'  # Replace 'your_image_path.jpg' with the path to your image
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print("Error: Unable to load image.")
else:
    # Specify the number of rows and columns for the grid
    rows = 20
    cols = 10

    # Draw the grid
    image_with_grid = draw_grid(image.copy(), rows, cols)

    # Display the image with grid
    cv2.imshow('Image with Grid', image_with_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

