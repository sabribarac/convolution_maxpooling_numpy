from PIL import Image
import numpy as np

def apply_convolution(img:np.array, kernel:np.array):
    
    # Get the height, width, and number of channels of the image
    height,width,c =img.shape[0],img.shape[1],img.shape[2]
    
    # Get the height, width, and number of channels of the kernel
    kernel_height,kernel_width = kernel.shape[0],kernel.shape[1]
    
    # Create a new image of the size minus the border 
    # where the convolution can't be applied
    new_img = np.zeros((height-kernel_height+1,width-kernel_width+1,3)) 
    
    # Loop through each pixel in the image
    for i in range(kernel_height//2, height-kernel_height//2-1):
        for j in range(kernel_width//2, width-kernel_width//2-1):
            # Extract a window of pixels around the current pixel
            window = img[i-kernel_height//2 : i+kernel_height//2+1,j-kernel_width//2 : j+kernel_width//2+1]
            
            # Apply the convolution to the window and set the result as the value of the current pixel in the new image
            new_img[i, j, 0] = int((window[:,:,0] * kernel).sum())
            new_img[i, j, 1] = int((window[:,:,1] * kernel).sum())
            new_img[i, j, 2] = int((window[:,:,2] * kernel).sum())
      
    # Clip values to the range 0-255
    new_img = np.clip(new_img, 0, 255)
    
    # Create a PIL image from the new image and display it
    sImg = Image.fromarray(new_img.astype(np.uint8))
    sImg.show()

if __name__ == "__main__":
    # Kernel to blur image
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])*1/16
    
    # kernel to get the edges of an image
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    # Import image through PIL and convert to numpy array
    img = Image.open('city.jpg')
    or_img = np.asarray(img)
  
    apply_convolution(or_img, kernel)    
