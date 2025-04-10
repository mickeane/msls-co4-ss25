import cv2 as cv
import numpy as np

### TODO: INTRODUCE THIS FUNCTION INTO THE TUTORIAL ON COLOR SPACES

if False:
    # TODO: THIS SHOULD GO INTO THE COLOR SPACE TUTORIAL
    # Note: imageB should be an image with alpha channel
    grayB = convert_with_alpha(imageB, code=cv.COLOR_BGRA2GRAY)
    grayB2 = cv.cvtColor(imageB, cv.COLOR_BGRA2GRAY)
    show_image_chain((grayB, "gray"), (grayB2, "gray Opencv"))



def convert_with_alpha(image, 
                       code=cv.COLOR_BGR2GRAY, 
                       background=255):
    """Apply a color conversion while preserving the information in the
    alpha channel. Assumption: The alpha is at the fourth channel.
    
    Note: No intensive testing of the input is performed.

    Args: 
        image: The image to convert.
        code:  The color space conversion code.
        background: The background color to use for the alpha channel.
    """
    assert image.dtype == np.uint8, "Input image does not have 8-bit depth"

    if len(image.shape) == 2:
        # Already a grayscale image without alpha channel.
        return image
    elif image.shape[2] != 4:
        # Image without alpha channel. Perform a normal conversion.
        return cv.cvtColor(image, code)
    else:
        # Split the image into color and alpha channels.
        color = image[:, :, :3]
        alpha = image[:, :, 3].astype(float) / 255.

        # Convert the color channels.
        foreground = cv.cvtColor(color, code)

        # Apply alpha channel.
        is_fg_gray = len(foreground.shape) == 2
        is_bg_gray = isinstance(background, (int, float))

        if is_fg_gray and is_bg_gray:
            # Output: Grayscale
            ret = foreground * alpha + background * (1-alpha)
        else:
            # Output: Color
            if is_fg_gray:
                foreground = foreground * alpha
                foreground = np.stack([foreground, foreground, foreground], axis=2,)
            else:
                foreground = foreground * alpha[..., np.newaxis]

            if is_bg_gray:
                background = np.array([background, background, background])
            background = np.array(background) * (1-alpha)[..., np.newaxis]
            ret = foreground + background
        ret = np.clip(ret, 0, 255).astype(np.uint8)
        return ret
