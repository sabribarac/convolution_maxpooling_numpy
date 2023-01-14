from PIL import Image
import numpy as np

class MaxPool2D():
    """
    Applies 2D max pooling to the input tensor.
    
    Args:
        inputs: a 3D NumPy array with dimensions (batch size, height, width, channels).
        pool_size: a tuple or integer specifying the size of the pooling window.
        strides: a tuple or integer specifying the strides of the pooling operation.
    
    Returns:
        a 4D NumPy array with dimensions (batch size, new_height, new_width, channels),
        where new_height and new_width are computed as:
        new_height = (height - pool_height) // stride_height + 1
        new_width = (width - pool_width) // stride_width + 1
    """
    def __init__(self, pool_size=(2,2), strides=None, padding='valid') -> None:
        self.pool_size = tuple(pool_size) if isinstance(pool_size, tuple) else (pool_size,) * 2
        if not isinstance(strides, tuple) and strides is not None:
            strides = (strides,strides) if isinstance(strides, int) else tuple(strides)
        self.strides = strides or self.pool_size
        if padding.lower() not in {'valid','same'}:
            raise ValueError(
                'Padding must be valid or same but received: {padding}'
            )
        self.padding = padding
    
    

    def __call__(self,inputs):
        pool_height, pool_width = self.pool_size
        strides, _ = self.strides
        height, width, channels = inputs.shape
        m_height = (height - pool_height) // strides + 1
        m_width = (width - pool_width) // strides + 1
        outputs = np.empty((m_height, m_width, channels))
        
        for h in range(m_height):
            for w in range(m_width):
                window = inputs[h*strides:h*strides+pool_height, w*strides:w*strides+pool_width, :]
                outputs[h, w, :] = np.amax(window, axis=(0, 1))
        
        return outputs.astype(np.uint8)
    


if __name__ == '__main__':
    img = Image.open('kitten3.jpg')
    or_img = np.asarray(img)
    sImg = Image.fromarray(or_img)
    sImg.show()
    print(or_img.shape)

    pool_img = MaxPool2D(pool_size=2,strides=2)(or_img)
    print(pool_img.shape)

    sImg = Image.fromarray(pool_img)
    sImg.show()
    exit()
