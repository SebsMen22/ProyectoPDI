"""
diagnostico_rendimiento.py - mide en TU computador cuanto tarda cada etapa de un cuadro y dice cual frena.

Etapas: leer el cuadro, reducirlo, MediaPipe (deteccion), clasificador, OSC, dibujo y JPEG.
Uso (desde la carpeta del proyecto, con el entorno activado):
    python src\\diagnostico_rendimiento.py --video INTERFAZ\\videos\\usuarios\\Mariana_ambas.mp4
    python src\\diagnostico_rendimiento.py --camara 1
Recomendacion: correrlo 2 veces, una con TouchDesigner y Edge cerrados y otra con todo abierto, para ver
cuanto te quita la competencia por el procesador.
"""
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "INTERFAZ"))
import pose_osc as P                       # noqa: E402
from postura import ClasificadorVivo       # noqa: E402
from pythonosc.udp_client import SimpleUDPClient  # noqa: E402
import motor as M                          # noqa: E402


def main():
    # --- 1. opciones: video o camara, y cuantos cuadros medir ---
    ap = argparse.ArgumentParser(description="Mide cuanto tarda cada etapa del procesamiento de un cuadro.")
    ap.add_argument("--video", default=None)
    ap.add_argument("--camara", type=int, default=1)
    ap.add_argument("--cuadros", type=int, default=150)
    a = ap.parse_args()
    # --- 2. abrir la fuente, MediaPipe, el clasificador y el cliente OSC (los mismos que usa la interfaz) ---
    cap = cv2.VideoCapture(a.video) if a.video else cv2.VideoCapture(a.camara, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("No pude abrir la fuente.")
    op = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=P.buscar_modelo()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    marcador = mp.tasks.vision.PoseLandmarker.create_from_options(op)
    ruta = os.path.join(AQUI, "..", "modelos", "postura.joblib")
    clasif = ClasificadorVivo(ruta, 5, 3) if os.path.isfile(ruta) else None
    cli = SimpleUDPClient("127.0.0.1", 7000)
    # --- 3. medir: T acumula los segundos de cada etapa; luego se divide entre los cuadros ---
    T = {k: 0.0 for k in ("leer", "reducir", "mediapipe", "clasificador", "osc", "dibujo", "jpeg")}
    n, t0, prev = 0, time.time(), {}
    while n < a.cuadros:
        t_a = time.perf_counter()
        ok, fr = cap.read()                      # etapa: leer (en video incluye decodificar)
        if not ok:
            break
        t_b = time.perf_counter(); T["leer"] += t_b - t_a
        if fr.shape[1] > 640:                    # etapa: reducir a 640 px de ancho, como la interfaz
            fr = cv2.resize(fr, (640, int(fr.shape[0] * 640 / fr.shape[1])), interpolation=cv2.INTER_AREA)
        vista = fr.copy()
        t_c = time.perf_counter(); T["reducir"] += t_c - t_b
        # etapa: MediaPipe (la mas pesada)
        res = marcador.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)),
                                        int((time.time() - t0) * 1000))
        t_d = time.perf_counter(); T["mediapipe"] += t_d - t_c
        t_h = t_d
        if res.pose_landmarks:
            pts = res.pose_landmarks[0]
            b = p = 0
            if clasif:                           # etapa: clasificador de postura
                b, p, _ = clasif.actualizar([(q.x, q.y) for q in pts])
            t_e = time.perf_counter(); T["clasificador"] += t_e - t_d
            # etapa: OSC (los 4 canales generales + 12 articulaciones x 3 valores)
            for canal in ("/detected", "/posture", "/arms_open", "/legs_open"):
                cli.send_message(canal, 1.0)
            for nombre, idx in P.JOINTS.items():
                q = pts[idx]
                x, y, v = P.suavizar_y_velocidad(prev, nombre, q.x, q.y, 0.03, P.ALPHA)
                for suf, val in (("x", x), ("y", y), ("s", v)):
                    cli.send_message(f"/{nombre}_{suf}", float(val))
            t_f = time.perf_counter(); T["osc"] += t_f - t_e
            # etapa: dibujo del esqueleto (se reutiliza el dibujo de la interfaz sin crear un motor completo)
            m = M.Motor.__new__(M.Motor)
            M.Motor._dibujar(m, vista, pts, False, bool(b), bool(p))
            t_h = time.perf_counter(); T["dibujo"] += t_h - t_f
        cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 80])      # etapa: JPEG (lo que ve el navegador)
        T["jpeg"] += time.perf_counter() - t_h
        n += 1
    # --- 4. resultados: ms por cuadro de cada etapa y cual es la que mas tarda ---
    total = (time.time() - t0) / max(1, n) * 1000
    print(f"\n{n} cuadros - {total:.1f} ms por cuadro en total ({1000 / total:.0f} FPS)")
    for k, v in T.items():
        print(f"  {k:13s} {v / max(1, n) * 1000:6.1f} ms")
    mayor = max(T, key=T.get)
    print(f"\nLo que mas tarda: {mayor}.")
    if mayor == "mediapipe":
        print("  -> Cierra TouchDesigner/Edge y prueba de nuevo; usa modo 'Maximo rendimiento' y el cargador.")
    elif mayor == "leer":
        print("  -> Si es video: reducelo con src\\optimizar_video.py. Si es camara: falta luz o hay otra app usandola.")


if __name__ == "__main__":
    main()
