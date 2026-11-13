"""
listar_camaras.py  -  Shadow Dance
Encuentra el número (índice) de tu cámara USB.

Windows no le da nombre a las cámaras en OpenCV: solo las numera 0, 1, 2...
Este programa prueba cada número y muestra la imagen en una ventana, para que
veas cuál es tu "USB2.0 HD UVC WebCam" (truco: tapa el lente con el dedo y mira
cuál ventana se oscurece).

Uso:
    python src/listar_camaras.py
En la ventana: cualquier tecla = pasar a la siguiente cámara, q = salir.
"""
import cv2

# DirectShow es el "controlador" de video de Windows que mejor funciona con webcams USB.
CAP_DSHOW = getattr(cv2, "CAP_DSHOW", 700)
MAX_INDICES = 6  # se prueban los números 0 a 5


def main():
    encontradas = []
    for i in range(MAX_INDICES):
        cap = cv2.VideoCapture(i, CAP_DSHOW)
        if not cap.isOpened():
            print(f"Índice {i}: no hay cámara")
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok:
            print(f"Índice {i}: se abrió, pero no entrega imagen")
            cap.release()
            continue

        alto, ancho = frame.shape[:2]
        print(f"Índice {i}: funciona ({ancho}x{alto}). Mírala en la ventana.")
        encontradas.append(i)

        tecla = 255  # 255 significa "ninguna tecla pulsada"
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"Indice {i}  ({ancho}x{alto})  -  tecla: siguiente, q: salir",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camaras disponibles", frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla != 255:
                break
        cap.release()
        if tecla == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Cámaras que funcionan:", encontradas)
    if encontradas:
        print("Usa el índice de tu cámara USB así:  python src/pose_osc.py --camara N")


if __name__ == "__main__":
    main()
