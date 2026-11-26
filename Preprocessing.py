from PIL import Image
import numpy as np
import cv2

def main(): #kod ma na celu przeprowadzić preprocessing obrazów do późniejszego użycia do trenowania modelu
    for i in range (1,101):
        image = np.asarray(Image.open(f"crossings/crossing_{i}.jpg")) #wczytuje obraz i przetwarza go na array
        image_min_max = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX) #przeprowadza normalizację obrazu (min-max scaling)
        (Image.fromarray(image_min_max)).save(f"crossings_preprocessed/crossing_preprocessed_{i}.jpg") #zapisuje obraz pod nową nazwą do drugiego folderu

if __name__ == "__main__":
    main()