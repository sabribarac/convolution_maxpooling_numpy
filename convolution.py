from PIL import Image
import numpy as np

def apply_convolution(img:np.array, kernel:np.array):
    
    # Get the height, width, and number of channels of the image
    height,width,c =img.shape[0],img.shape[1],img.shape[2]
    
    # Get the height, width, and number of channels of the kernel
    kernel_height,kernel_width = kernel.shape[0],kernel.shape[1]
    
    # Create a new image of original img size minus the border 
    # where the convolution can't be applied
    new_img = np.zeros((height-kernel_height+1,width-kernel_width+1,3)) 
    
    # Loop through each pixel in the image
    # But skip the outer edges of the image
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
    return new_img.astype(np.uint8)

if __name__ == "__main__":

    # kernel for edge detection
    kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    
    # kernel for vertical edge detection
    #kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    # kernel for horizontal edge detection
    # kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    # Kernel for box blur 
    # kernel = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])

    # Open the image and convert it to an array
    # Try to put your own picture!
    img = Image.open('kitten3.jpg')
    or_img = np.asarray(img)
    
    new_img = apply_convolution(or_img, kernel)

    # Create a PIL image from the new image and display it     
    sImg = Image.fromarray(new_img)
    sImg.show() 
