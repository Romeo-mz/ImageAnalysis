import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_operations.thinning import zhang_suen

def threshold(img, seuil):
    img[img < seuil] = 0 #Tous les pixels en dessous du seuil deviennent 0
    img[img >= seuil] = 255 #Tous les pixels au dessus du seuil deviennent 255
    return img


def addition(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    if img1.shape != img2.shape:
        height, width = img1.shape
        img2 = cv2.resize(img2, (width, height))

    result = np.clip(img1 + img2, 0, 255)
    return result


def subtraction(img1, img2):
    if img1.shape != img2.shape:
        height, width = img1.shape
        img2 = cv2.resize(img2, (width, height))

    result = np.clip(img1 - img2, 0, 255)
    return result


def erosion(img, radius):
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            count = np.sum(img[max(0, i - radius):min(img.shape[0], i + radius + 1),
                             max(0, j - radius):min(img.shape[1], j + radius + 1)] > 0)

            if count == (2 * radius + 1) ** 2:
                result[i, j] = 255

    return result


def dilation(img, kernel_size=3):
    result = img.copy()
    padding = kernel_size // 2

    for i in range(padding, img.shape[0] - padding):
        for j in range(padding, img.shape[1] - padding):
            window = img[i - padding: i + padding + 1, j - padding: j + padding + 1]
            if np.all(window == 0):
                result[i, j] = 0
            else:
                result[i, j] = 255

    return result


def opening(img, kernel_size=3):
    return dilation(erosion(img, kernel_size), kernel_size)


def closing(img, kernel_size=3):
    return erosion(dilation(img, kernel_size), kernel_size)


def thinning(img, iterations=1):
    result = img.copy()

    for _ in range(iterations):
        result = np.logical_and(result, erosion(result, 3)) #Les pixels qui se chevauchent deviennent 0
    
    return result



def thickening(binary, iterations=1):
    result = binary.copy()
    kernel_size = 2

    for _ in range(iterations):
        result = np.logical_or(result, dilation(result, kernel_size)) #Les pixels qui se chevauchent deviennent 1

    return result


def lantuejoul_skeletonization(img):
    out = np.zeros_like(img)
    before = np.zeros_like(img)
    n = 0
    max_iterations = 1

    while True:
        before = np.copy(out)

        eroded = erosion(img, n)
        opened = opening(eroded, 1)
        sub = subtraction(eroded, opened)

        out = addition(out, sub)

        if np.array_equal(out, before):
            print("Converged!")
            break

        n += 1
        if n >= max_iterations:
            print("Max iterations reached!")
            break

    return out


def homotopic_skeletonization(img):
    before = np.copy(img)
    out = zhang_suen(img)

    while not np.array_equal(out, before):
        before = np.copy(out)
        out = zhang_suen(out)

    return out

