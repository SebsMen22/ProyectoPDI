"""
grabar_video.py - Graba los videos de prueba de un usuario con la MISMA camara y resolucion del sistema.
================================================================================================
Por cada usuario graba 4 videos cortos (quieto, brazos, piernas, ambas). Sirven para medir el prototipo con
usuarios de prueba: despues se pasan a evaluar_sistema.py, que cuenta el % de cuadros con la postura correcta.

Uso (un usuario a la vez):
    python src\\grabar_video.py --usuario ana --camara 1 --sin-mjpg --ancho 640 --alto 480

Teclas:  ESPACIO = empezar la grabacion de la postura mostrada (cuenta de 3 s)   s = saltar esa postura   q = salir
Los videos quedan en INTERFAZ\\videos\\usuarios\\ana_quieto.mp4, ana_brazos.mp4, ana_piernas.mp4, ana_ambas.mp4
(se guardan SIN efecto espejo, como los entregaria la camara; la vista previa si es espejo para que sea comoda).
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import cv2

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import pose_osc as P          # reutiliza abrir_fuente: mismo formato de camara que el sistema en vivo

# (nombre del archivo, texto en pantalla, clase que espera evaluar_sistema.py)
POSTURAS = [
    ("quieto", "DE PIE, BRAZOS Y PIERNAS CERRADOS", "ninguna"),
    ("brazos", "ABRE LOS BRAZOS (piernas cerradas)", "brazos"),
    ("piernas", "ABRE LAS PIERNAS (brazos abajo)", "piernas"),
    ("ambas", "ABRE BRAZOS Y PIERNAS", "ambas"),
]


def texto(img, s, y, tam=0.8, color=(255, 255, 255), grosor=2):
    """Texto con sombra negra para que se lea sobre cualquier fondo."""
    cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, tam, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, tam, color, grosor, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Graba los 4 videos de prueba de un usuario.")
    ap.add_argument("--usuario", required=True, help="nombre corto del usuario (sin espacios), ej. ana")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", default="dshow")
    ap.add_argument("--sin-mjpg", action="store_true")
    ap.add_argument("--segundos", type=float, default=10.0, help="duracion de cada video")
    ap.add_argument("--carpeta", default=os.path.join("INTERFAZ", "videos", "usuarios"))
    a = ap.parse_args()

    os.makedirs(a.carpeta, exist_ok=True)
    cap = P.abrir_fuente(SimpleNamespace(video=None, camara=a.camara, backend=a.backend, sin_mjpg=a.sin_mjpg,
                                         ancho=a.ancho, alto=a.alto, fps=a.fps))
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ventana = "Shadow Dance - grabar usuario"
    hechos = []

    for nombre, instruccion, clase in POSTURAS:
        ruta = os.path.join(a.carpeta, f"{a.usuario}_{nombre}.mp4")
        # --- espera: se muestra la instruccion hasta que alguien pulse ESPACIO ---
        listo = saltar = False
        while not listo and not saltar:
            ok, frame = cap.read()
            if not ok:
                continue
            vista = cv2.flip(frame, 1)
            texto(vista, f"{a.usuario}: {instruccion}", 32, 0.7, (80, 220, 255))
            texto(vista, "ESPACIO = grabar   s = saltar   q = salir", alto - 14, 0.6)
            cv2.imshow(ventana, vista)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return finalizar(a, hechos)
            listo, saltar = k == 32, k == ord("s")
        if saltar:
            continue

        # --- cuenta regresiva de 3 s ---
        fin_cuenta = time.time() + 3
        while time.time() < fin_cuenta:
            ok, frame = cap.read()
            if not ok:
                continue
            vista = cv2.flip(frame, 1)
            texto(vista, str(int(fin_cuenta - time.time()) + 1), alto // 2, 3.0, (0, 0, 255), 6)
            texto(vista, instruccion, 32, 0.7, (80, 220, 255))
            cv2.imshow(ventana, vista)
            cv2.waitKey(1)

        # --- grabacion: se escribe el cuadro original (sin espejo) ---
        salida = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (ancho, alto))
        inicio, cuadros = time.time(), 0
        while time.time() - inicio < a.segundos:
            ok, frame = cap.read()
            if not ok:
                continue
            salida.write(frame)
            cuadros += 1
            vista = cv2.flip(frame, 1)
            texto(vista, f"GRABANDO {nombre}  {a.segundos - (time.time() - inicio):.0f} s", 32, 0.8, (0, 0, 255))
            cv2.imshow(ventana, vista)
            cv2.waitKey(1)
        salida.release()
        hechos.append((ruta, clase))
        print(f"Guardado {ruta}  ({cuadros} cuadros, {cuadros / a.segundos:.0f} FPS reales)")

    cap.release()
    cv2.destroyAllWindows()
    finalizar(a, hechos)


def finalizar(a, hechos):
    """Imprime el comando para medir los videos grabados."""
    if not hechos:
        print("No se grabo ningun video.")
        return
    lista = " ".join(f"{r}:{c}" for r, c in hechos)
    print("\nPara medir este usuario:\n  python src\\evaluar_sistema.py --videos " + lista)


if __name__ == "__main__":
    main()
