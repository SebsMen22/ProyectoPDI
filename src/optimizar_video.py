"""
optimizar_video.py - crea una copia LIVIANA de un video para usarlo en la pestana Pruebas de la interfaz.

Por que: un video de celular suele ser 4K (3840x2160). Solo decodificarlo gasta mucho tiempo por cuadro y la
interfaz baja a ~17 FPS aunque la deteccion sea rapida. La copia se reduce a 960 px de ancho (la interfaz
igual trabaja a 640 px), conserva los 30 FPS y aplica la rotacion que el celular guarda en el archivo.
El original no se toca: guardalo aparte para el video final.

Uso:
    python src\\optimizar_video.py INTERFAZ\\videos\\usuarios\\Test1.mp4
    python src\\optimizar_video.py video.mp4 --ancho 960 --salida otra_carpeta\\video_lite.mp4
Resultado por defecto: al lado del original, con el sufijo _lite.mp4
"""
import argparse
import os
import sys

import cv2


def main():
    ap = argparse.ArgumentParser(description="Crea una copia liviana de un video (menos pixeles, mismos FPS).")
    ap.add_argument("video", help="ruta del video original")
    ap.add_argument("--ancho", type=int, default=960, help="ancho de la copia en pixeles (por defecto 960)")
    ap.add_argument("--salida", default=None, help="ruta de la copia (por defecto <nombre>_lite.mp4)")
    a = ap.parse_args()

    if not os.path.isfile(a.video):
        sys.exit(f"No encuentro el video: {a.video}")
    base, _ = os.path.splitext(a.video)
    salida = a.salida or base + "_lite.mp4"

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit("No pude abrir el video (revisa que sea MP4).")
    # que OpenCV aplique la rotacion guardada por el celular, si su version lo permite
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    giro = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)          # 0, 90, 180 o 270
    auto = int(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) or 0)          # 1 si OpenCV ya la aplica
    rotar = None
    if giro and not auto:                                             # versiones viejas: girar a mano
        rotar = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(giro)

    escritor, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rotar is not None:
            frame = cv2.rotate(frame, rotar)
        if escritor is None:
            alto = int(round(frame.shape[0] * a.ancho / frame.shape[1]))
            alto -= alto % 2
            escritor = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (a.ancho, alto))
        escritor.write(cv2.resize(frame, (a.ancho, alto), interpolation=cv2.INTER_AREA))
        n += 1
        if n % 60 == 0:
            print(f"  {n}/{total or '?'} cuadros", end="\r")
    cap.release()
    if escritor is not None:
        escritor.release()
    mb = os.path.getsize(salida) / 1e6 if os.path.exists(salida) else 0
    print(f"\nListo: {salida}  ({n} cuadros, {a.ancho}x{alto}, {fps:.0f} FPS, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
