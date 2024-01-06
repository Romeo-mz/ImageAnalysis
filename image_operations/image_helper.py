import cv2
from matplotlib import pyplot as plt

def read_image(path):
    try:
        return cv2.imread(path, 0)
    except Exception as e:
        print(f"Error reading the image: {e}")
        return None

def show_image(img):
    try:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error displaying image: {e}")

def show_multiple_images(images, titles):
    try:
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(10 * num_images, 10))

        for i in range(num_images):
            axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            axes[i].set_title(titles[i])
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error displaying multiple images: {e}")

    constL = [
    [255, 255, 255],
    [None, 255, None],
    [0, 0, 0]
    ]

    constM = [
        [255, 255, None],
        [255, 255, 0],
        [None, 0, 0]
    ]
