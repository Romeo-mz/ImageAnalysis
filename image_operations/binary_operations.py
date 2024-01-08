import cv2
import matplotlib.pyplot as plt
import numpy as np
from image_operations.thinning import zhang_suen

def threshold(img, seuil):
    """
    Applique un seuillage sur l'image en convertissant les pixels en noir ou blanc en fonction d'un seuil donné.
    Les pixels inférieurs au seuil deviennent noirs (0) et les pixels supérieurs ou égaux au seuil deviennent blancs (255).
    """
    img[img < seuil] = 0 
    img[img >= seuil] = 255 
    return img


def addition(img1, img2):
    """
    Effectue l'addition pixel à pixel de deux images.
    Les valeurs des pixels sont clipées entre 0 et 255 pour éviter les dépassements.
    """
    img1 = np.array(img1)
    img2 = np.array(img2)

    if img1.shape != img2.shape:
        height, width = img1.shape
        img2 = cv2.resize(img2, (width, height))

    result = np.clip(img1 + img2, 0, 255)
    return result


def subtraction(img1, img2):
    """
    Effectue la soustraction pixel à pixel de deux images.
    Les valeurs des pixels sont clipées entre 0 et 255 pour éviter les dépassements.
    """
    if img1.shape != img2.shape:
        height, width = img1.shape
        img2 = cv2.resize(img2, (width, height))

    result = np.clip(img1 - img2, 0, 255)
    return result


def erosion(img, radius):
    """
    Effectue une érosion morphologique sur l'image en utilisant un élément structurant de taille (2 * radius + 1) x (2 * radius + 1).
    Un pixel de l'image de sortie est blanc (255) si tous les pixels de l'élément structurant correspondants sont blancs (255) dans l'image d'entrée.
    """
    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            count = np.sum(img[max(0, i - radius):min(img.shape[0], i + radius + 1),
                             max(0, j - radius):min(img.shape[1], j + radius + 1)] > 0)

            if count == (2 * radius + 1) ** 2:
                result[i, j] = 255

    return result


def dilation(img, kernel_size=3):
    """
    Effectue une dilatation morphologique sur l'image en utilisant un élément structurant de taille kernel_size x kernel_size.
    Un pixel de l'image de sortie est blanc (255) si au moins un pixel de l'élément structurant correspondant est blanc (255) dans l'image d'entrée.
    """
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
    """
    Effectue une ouverture morphologique sur l'image en utilisant une érosion suivie d'une dilatation.
    Cela permet de supprimer les petits objets et les détails fins tout en préservant la forme globale des objets plus grands.
    """
    return dilation(erosion(img, kernel_size), kernel_size)


def closing(img, kernel_size=3):
    """
    Effectue une fermeture morphologique sur l'image en utilisant une dilatation suivie d'une érosion.
    Cela permet de remplir les petits trous et les fissures tout en préservant la forme globale des objets.
    """
    return erosion(dilation(img, kernel_size), kernel_size)


def thinning(img, iterations=1):
    """
    Effectue un amincissement morphologique sur l'image en utilisant l'algorithme de Zhang-Suen.
    Les pixels qui se chevauchent deviennent noirs (0).
    """
    result = img.copy()

    for _ in range(iterations):
        result = np.logical_and(result, erosion(result, 3))
    
    return result


def thickening(binary, iterations=1):
    """
    Effectue un épaississement morphologique sur l'image binaire en utilisant une dilatation.
    Les pixels qui se chevauchent deviennent blancs (255).
    """
    result = binary.copy()
    kernel_size = 2

    for _ in range(iterations):
        result = np.logical_or(result, dilation(result, kernel_size))

    return result


def lantuejoul_skeletonization(img):
    """
    Effectue une squelettisation de Lantuejoul sur l'image binaire en utilisant des opérations morphologiques.
    La squelettisation est itérativement appliquée jusqu'à convergence.
    """
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
    """
    Effectue une squelettisation homotopique sur l'image binaire en utilisant l'algorithme de Zhang-Suen.
    La squelettisation est itérativement appliquée jusqu'à convergence.
    """
    before = np.copy(img)
    out = zhang_suen(img)

    while not np.array_equal(out, before):
        before = np.copy(out)
        out = zhang_suen(out)

    return out
