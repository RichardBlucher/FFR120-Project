import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def image_to_matrix(image_path, matrix_size=(300, 300)):
    
    img = cv2.imread(image_path)

    # 550x550 before resize
    # Resize the image to 200x200
    img2 = cv2.resize(img, matrix_size)

    # Initialize the matrix with zeros (sea)
    matrix = np.zeros((img2.shape[0], img2.shape[1]))

    # Iterate through the image to fill the matrix
    for y in range(img2.shape[0]):
        for x in range(img2.shape[1]):
            # Get the pixel color (BGR format from OpenCV)
            pixel = img2[y, x]
            
            blue, green, red = pixel

            # Check if the pixel is red (food) by detecting if red is dominant
            #if red > 200 and green < 100 and blue < 100:  # Threshold for red color
                #matrix[y, x] = 2  
            if red == 0 and green == 0 and blue == 0:  # white pixel (land)
                matrix[y, x] = 1
            else: # black pixel (sea)
                matrix[y, x] = 0

    boundrylist = np.argwhere(matrix != 0)
    boundrydict = {(i,j) : matrix[i][j] for i,j in boundrylist}

    return boundrydict

image_path = r"../MapRef/Tokyo.png"  # path
matrix = image_to_matrix(image_path)




# Check if it finds/identifies the right amount of food locations 
def check_food(matrix):
    
    food_count = 0  
    found_food = False
  
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            if matrix[y, x] == 2:  
                print(f"Food found at ({x}, {y})")
                found_food = True
                food_count += 1
    
    if not found_food:
        print("No food found in the map.")
    
    return food_count  


cmap = ListedColormap(['black', 'white'])

plt.imshow(matrix, cmap=cmap, interpolation='nearest')
plt.colorbar(label='Map Values')
plt.title('Matrix')
plt.show()
    


