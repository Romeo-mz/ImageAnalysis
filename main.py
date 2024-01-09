import cv2
from matplotlib import pyplot as plt
import image_operations.binary_operations as bo
from image_operations.thinning import zhang_suen
from image_operations.thickening import zhang_suen_thicken
from image_operations.image_helper import read_image

rel_path = "images/"
img1 = read_image(rel_path + 'test.jpeg')
img2 = read_image(rel_path + 'text.png')

def main():
    fig = plt.figure(figsize=(10, 7))
    rows = 3
    columns = 4
    titles = ['Original Image', 'Thresholded Image', 'Eroded Image', 'Dilated Image']

    # Reading the original image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title(titles[0])

    # Reading the copy image
    # fig.add_subplot(rows, columns, 13)
    # plt.imshow(img2)
    # plt.axis('off')
    # plt.title('Copy Image')
    
     # Applying threshold
    threshold = 127
    threshold_img = bo.threshold(img1, threshold)
    fig.add_subplot(rows, columns, 4)
    plt.imshow(threshold_img, cmap='gray')
    plt.title(titles[1])
    plt.xlabel('Threshold: {}'.format(threshold))  # Parameter explanation

    # Addition
    addition_img = bo.addition(img1, img2)
    fig.add_subplot(rows, columns, 2)
    plt.imshow(addition_img, cmap='gray')
    plt.axis('off')
    plt.title('Addition Image')

    # Subtraction
    subtraction_img = bo.subtraction(img1, img2)
    fig.add_subplot(rows, columns, 3)
    plt.imshow(subtraction_img, cmap='gray')
    plt.axis('off')
    plt.title('Subtraction Image')


    # Applying erosion
    erosion_img = bo.erosion(threshold_img, 3)
    fig.add_subplot(rows, columns, 5)
    plt.imshow(erosion_img, cmap='gray')
    plt.title(titles[2])
    plt.xlabel('Kernel Size: 3')  # Parameter explanation

    plt.tight_layout()

    # Applying dilation
    dilation_img = bo.dilation(threshold_img, 3)
    fig.add_subplot(rows, columns, 6)
    plt.imshow(dilation_img, cmap='gray')
    plt.title(titles[3])
    plt.xlabel('Kernel Size: 3')  # Parameter explanation
    
    plt.tight_layout()

    # Applying opening
    opening_img = bo.opening(threshold_img, 3)
    fig.add_subplot(rows, columns, 11)
    plt.imshow(opening_img, cmap='gray')
    plt.title('Opening')
    plt.xlabel('Kernel Size: 3')  # Parameter explanation

    # Applying closing
    closing_img = bo.closing(threshold_img, 3)
    fig.add_subplot(rows, columns, 12)
    plt.imshow(closing_img, cmap='gray')
    plt.title('Closing')
    plt.xlabel('Kernel Size: 3')  # Parameter explanation

    # Applying thinning
    _, binary_img = cv2.threshold(threshold_img, 127, 1, cv2.THRESH_BINARY)
    thinning_img = zhang_suen(binary_img, 10)
    fig.add_subplot(rows, columns, 7)
    plt.imshow(thinning_img, cmap='gray')
    plt.title('Thinning')
    plt.xlabel('Iterations: 10')  # Parameter explanation

    plt.tight_layout()

    # Applying thickening
    thickening_img = zhang_suen_thicken(binary_img, 7)
    fig.add_subplot(rows, columns, 8)
    plt.imshow(thickening_img, cmap='gray')
    plt.title('Thickening')
    plt.xlabel('Iterations: 7')  # Parameter explanation

    plt.tight_layout()

    # Applying skeletonization by Lantuejoul
    skeletonization_img = bo.lantuejoul_skeletonization(binary_img)
    fig.add_subplot(rows, columns, 9)
    plt.imshow(skeletonization_img, cmap='gray')
    plt.axis('off')
    plt.title('Lantuejoul skeletonization')

    # Applying skeletonization homotecy L
    homotopic_skeletonization = bo.homotopic_skeletonization(binary_img)
    fig.add_subplot(rows, columns, 10)
    plt.imshow(homotopic_skeletonization, cmap='gray')
    plt.axis('off')
    plt.title('Homotopic skeletonization')


    plt.tight_layout()


if __name__ == "__main__":
    main()
    plt.show()
