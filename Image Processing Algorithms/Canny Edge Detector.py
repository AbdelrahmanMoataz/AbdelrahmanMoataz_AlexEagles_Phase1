import numpy as np


def conv(image, kernel):
    """ An implementation of convolution filter.
    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).
    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # Use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

   
    for i in range(Hk):
        for j in range(Wk):
            # Create a matrix of pixels same size as kernel.
            # np.multiply it with the kernel for element wise multiplication
            # then sum result for to find convolution at pixel (i,j)
            # This loop ensures that kernel is applied to the whole image.
            image_kernel = padded[i: i + Hk, j: j + Wk]
            out[i, j] = np.sum(np.multiply(image_kernel, kernel))
    

    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.
    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.
    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    
    # Gaussian Kernel formula for a kernel of size (2k + 1 x 2k +1) requires the
    # element position to be deducted by (k+1) eg: 5x5 kernel has  5 = 2k + 1
    # which implies k = (5 - 1)/2 = 2. k will always be whole number because size is odd
    k = (size - 1) / 2
    constant = 1 / (2.0 * np.pi * sigma ** 2)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = constant * np.exp(-((i - (k + 1) ** 2) + (j - (k + 1) ** 2)) / (2 * sigma ** 2))
    

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.
    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    
    # Using the Sobel edge detector:
    Sobel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    out = conv(img, Sobel_X)
    

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.
    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    
    # Using the Sobel edge detector:
    Sobel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    out = conv(img, Sobel_Y)
    

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    
    # Find the x and y edges of the image
    # G = square root of squared sum of each element
    # theta = arctan of y by x of each element
    Ix = partial_x(img)
    Iy = partial_y(img)
    sum_squared = np.square(Ix) + np.square(Iy)
    np.sqrt(sum_squared, out=G)  # output is directly to G, no need to assign again
    np.arctan2(Iy, Ix, out=theta)  # output is directly to G, no need to assign again
    

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)


    for i in range(H):
        for j in range(W):
            ## Checking gradient at angle 90 is same as checking at angle -90 (270)
            angle = theta[i, j]
            if angle >= 180:
                angle -= 180

            # Define neighboring pixels based on gradient angle
            if angle == 0:  # directly to left and right of pixel
                n1 = i, j - 1
                n2 = i, j + 1
            elif angle == 45:  # bottom left and top right
                n1 = i - 1, j - 1
                n2 = i + 1, j + 1
            elif angle == 90:  # down and up
                n1 = i - 1, j
                n2 = i + 1, j
            elif angle == 135:  # bottom right and top left
                n1 = i - 1, j + 1
                n2 = i + 1, j - 1
            neighbors = get_neighbors(i, j, H, W)  # List of valid neighbors

            """
            if n1[0] < 0 or n1[1] < 0 or n2[0] < 0 or n2[1] < 0:
                if (n1[0] < 0 or n1[1] < 0) and (n2[0] < 0 or n2[1] < 0):
                    out[i,j] = G[i,j]
                    continue  # If both n1 and n2 have negative pixel position, keep this pixel
            else:
                n1 = n2  # if one or the other is negative, make current pixel compare only to valid neighbor
            """
            # Compare the magnitude of current pixels to neighboring pixels
            if n1 in neighbors:  # Checks if first neighbor is valid
                if G[i, j] < G[n1]:
                    continue  # No need to check with other neighbor
            if n2 in neighbors:  # Checks if second neighbor is valid
                if G[i, j] > G[n2]:
                    out[i, j] = G[i, j]  # If stronger than both, keep the pixel
            else:
                out[i, j] = G[i, j]  # Keep corner pixels (if both n1 and n2 invalid)
            # Since out array is made up of zeros, skipping the current loop without updating an out pixel
            # is as if we zeroed the G pixel.

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.
    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    strong_edges = img > high
    weak_edges = low < img <= high

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).
    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)
    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    for i in range(H):
        for j in range(W):
            if strong_edges[i, j]:  # Only check neighbors of strong edges
                for k in get_neighbors(i, j, H, W):  # extract valid neighbors
                    if weak_edges[k]:  # if valid neighbor is true, make the weak edge strong
                        edges[i, j] = True

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.
    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threshold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    # Find the kernel using given parameters then proceed with convolution
    kernel = gaussian_kernel(kernel_size, sigma)
    conv_img = conv(img, kernel)
    # Find the gradient of the convoluted image to obtain the non-maximum suppressed image
    G, theta = gradient(conv_img)
    suppressed_img = non_maximum_suppression(G, theta)
    # Find the strong and weak edges from the non-maximum suppressed image, then link the strong and weak edges
    # to get the final result
    strong_edges, weak_edges = double_thresholding(suppressed_img, high, low)
    edge = link_edges(strong_edges, weak_edges)

    return edge
