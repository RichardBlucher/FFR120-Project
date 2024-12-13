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
    matrix = np.zeros((img.shape[0], img.shape[1]))

    # Iterate through the image to fill the matrix
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Get the pixel color (BGR format from OpenCV)
            pixel = img[y, x]
            
            blue, green, red = pixel

            # Check if the pixel is red (food) by detecting if red is dominant
            if red > 200 and green < 100 and blue < 100:  # Threshold for red color
                matrix[y, x] = 2  
            elif red == 255 and green == 255 and blue == 255:  # white pixel (land)
                matrix[y, x] = 1  
            elif red == 0 and green == 0 and blue == 0:  # black pixel (sea)
                matrix[y, x] = 0  

    return matrix

image_path = r"../MapRef/GothenburgCropedFood.png"  # path
matrix = image_to_matrix(image_path)

print("Matrix shape:", matrix.shape)  # should print (200, 200)


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


food_count = check_food(matrix)
print(f"Total food locations found: {food_count}")



# Plot and compare to original picture
cmap = ListedColormap(['black', 'white', 'red'])

plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Map Values')
plt.title('Matrix')
plt.show()
    


