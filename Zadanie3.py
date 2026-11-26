from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def main():
    # Kod użyty do treningu modelu:
    #
    # model = YOLO("yolov8n.pt") #inicjalizacja modelu
    # model.train( #trening modelu na przygotowanych danych
    #     data = "data/data.yaml",
    #     epochs = 50,
    #     batch = 16,
    #     name = "yolov8_crossing_detection"
    # )
    # results = model.val() #walidacja modelu
    # print(results) #wypisanie rezultatów walidacji

    image1 = cv2.normalize(np.asarray(Image.open("files_for_detection/image1.jpg")), None, 0, 255, cv2.NORM_MINMAX) #otwiera obrazy do użycia przez model, zamienia je na array'e i od razu przeprowadza min-max scaling
    image2 = cv2.normalize(np.asarray(Image.open("files_for_detection/image2.jpg")), None, 0, 255, cv2.NORM_MINMAX)
    image3 = cv2.normalize(np.asarray(Image.open("files_for_detection/image3.jpg")), None, 0, 255, cv2.NORM_MINMAX)
    image4 = cv2.normalize(np.asarray(Image.open("files_for_detection/image4.jpg")), None, 0, 255, cv2.NORM_MINMAX)
    image5 = cv2.normalize(np.asarray(Image.open("files_for_detection/image5.jpg")), None, 0, 255, cv2.NORM_MINMAX)
    image6 = cv2.normalize(np.asarray(Image.open("files_for_detection/image6.jpg")), None, 0, 255, cv2.NORM_MINMAX)

    fig, ax = plt.subplots(2,3, figsize=(10,7), dpi=100) #przygotowuje możliwość wyświetlenia kilku obrazów na raz
    ax[0,0].imshow(image1) #wczytuje obrazy do wyświetlenia
    ax[0,1].imshow(image2)
    ax[0,2].imshow(image3)
    ax[1,0].imshow(image4)
    ax[1,1].imshow(image5)
    ax[1,2].imshow(image6)

    for i in range(2): #usuwa wartości na osiach wykresów (obrazów)
        for j in range(3):
            ax[i,j].get_xaxis().set_ticks([])
            ax[i,j].get_yaxis().set_ticks([])
    ax[0,1].set_title("Obrazy wejściowe", fontdict={'fontsize': 20}, pad=45) #ustawia tytuł widoczny wewnątrz okna
    plt.show() #pokazuje obrazy

    model = YOLO("runs/detect/yolov8_crossing_detection-2/weights/best.pt") #wybiera własnoręcznie wytrenowany model
    detectionImg1 = cv2.cvtColor(model(source=cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))[0].plot(), cv2.COLOR_BGR2RGB) #przeprowadza wykrywanie przejść dla pieszych i zamienia wyniki wykrywania na array. By obraz wynikowy był prawidłowy, konieczna jest tu dwukrotna zmiana systemu kolorów
    detectionImg2 = cv2.cvtColor(model(source=cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))[0].plot(), cv2.COLOR_BGR2RGB)
    detectionImg3 = cv2.cvtColor(model(source=cv2.cvtColor(image3, cv2.COLOR_RGB2BGR))[0].plot(), cv2.COLOR_BGR2RGB)
    detectionImg4 = cv2.cvtColor(model(source=cv2.cvtColor(image4, cv2.COLOR_RGB2BGR))[0].plot(), cv2.COLOR_BGR2RGB)
    detectionImg5 = cv2.cvtColor(model(source=cv2.cvtColor(image5, cv2.COLOR_RGB2BGR))[0].plot(), cv2.COLOR_BGR2RGB)
    detectionImg6 = cv2.cvtColor(model(source=cv2.cvtColor(image6, cv2.COLOR_RGB2BGR))[0].plot(), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2,3, figsize=(10,7), dpi=100) #przygotowuje możliwość wyświetlenia kilku obrazów na raz
    ax[0,0].imshow(detectionImg1) #wczytuje obrazy do wyświetlenia
    ax[0,1].imshow(detectionImg2)
    ax[0,2].imshow(detectionImg3)
    ax[1,0].imshow(detectionImg4)
    ax[1,1].imshow(detectionImg5)
    ax[1,2].imshow(detectionImg6)

    for i in range(2): #usuwa wartości na osiach wykresów (obrazów)
        for j in range(3):
            ax[i,j].get_xaxis().set_ticks([])
            ax[i,j].get_yaxis().set_ticks([])
    ax[0,1].set_title("Wykryte przejścia dla pieszych", fontdict={'fontsize': 20}, pad=50) #ustawia tytuł widoczny wewnątrz okna
    plt.show() #pokazuje obrazy


if __name__ == "__main__":
    main()