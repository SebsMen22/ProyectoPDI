"""
pose_osc.py  -  Shadow Dance
Cámara (o video) -> MediaPipe Pose -> posición y velocidad de cada articulación -> OSC a TouchDesigner.

Uso:
    python pose_osc.py            # webcam 0
    python pose_osc.py 1          # webcam 1 (si tienes más de una)
    python pose_osc.py video.mp4  # video grabado (útil para evaluar y repetir pruebas)
Salir: tecla q en la ventana de vista previa.

La vista previa dibuja las articulaciones: verde y grande = se mueve rápido (dejará estela), gris = quieta.
"""
import os
import sys
import time

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

# ---------------- Configuración ----------------
MODEL_NAME = "pose_landmarker_full.task"   # modelo descargado de la guía de MediaPipe Pose Landmarker
# Se busca junto a este script y, si no está, en la carpeta desde donde ejecutas el comando
_AQUI = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = next((r for r in (os.path.join(_AQUI, MODEL_NAME), MODEL_NAME) if os.path.exists(r)), None)
OSC_IP, OSC_PORT = "127.0.0.1", 7000       # TouchDesigner escucha aquí (OSC In CHOP)
MIRROR = True                              # True = efecto espejo (x -> 1 - x)
ALPHA = 0.5                                # suavizado EMA de la posición: menor = más suave, pero más retraso
MIN_VISIBILITY = 0.5                       # por debajo de esto la articulación no deja estela
UMBRAL_VISTA = 0.15                        # solo para la vista previa (velocidad que cuenta como "se mueve")

# Articulaciones que dejan estela (índices de MediaPipe Pose: 13/14 codos, 15/16 muñecas, 25/26 rodillas, 27/28 tobillos)
JOINTS = {
    "lw": 15, "rw": 16,
    "le": 13, "re": 14,
    "la": 27, "ra": 28,
    "lk": 25, "rk": 26,
}


def suavizar_y_velocidad(prev, nombre, x, y, dt):
    """Devuelve (x, y, velocidad) con suavizado EMA. Velocidad en 'anchos de imagen por segundo'."""
    if nombre in prev and dt:
        px, py = prev[nombre]
        x = ALPHA * x + (1 - ALPHA) * px
        y = ALPHA * y + (1 - ALPHA) * py
        vel = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 / dt
    else:
        vel = 0.0
    prev[nombre] = (x, y)
    return x, y, vel


def main(fuente=0):
    if MODEL_PATH is None:
        raise FileNotFoundError(
            f"No encuentro {MODEL_NAME}. Descárgalo de la guía de MediaPipe Pose Landmarker "
            "y ponlo junto a pose_osc.py o en la carpeta desde donde ejecutas el comando.")
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    client = SimpleUDPClient(OSC_IP, OSC_PORT)
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    prev = {}
    t_prev = None
    fps = 0.0
    t0 = time.time()

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time()
            dt = (t - t_prev) if t_prev else None
            t_prev = t
            if dt:
                fps = (1 / dt) if fps == 0 else 0.9 * fps + 0.1 * (1 / dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((t - t0) * 1000))

            vista = cv2.flip(frame, 1) if MIRROR else frame.copy()
            alto, ancho = vista.shape[:2]
            puntos_vista = []

            if result.pose_landmarks:
                client.send_message("/detected", 1.0)
                lms = result.pose_landmarks[0]
                for nombre, idx in JOINTS.items():
                    lm = lms[idx]
                    x = 1.0 - lm.x if MIRROR else lm.x
                    x, y, vel = suavizar_y_velocidad(prev, nombre, x, lm.y, dt)
                    if lm.visibility < MIN_VISIBILITY:
                        vel = 0.0
                    client.send_message(f"/{nombre}_x", float(x))
                    client.send_message(f"/{nombre}_y", float(y))
                    client.send_message(f"/{nombre}_s", float(vel))
                    puntos_vista.append((int(x * ancho), int(y * alto), vel))
            else:
                client.send_message("/detected", 0.0)   # sin bailarín: TouchDesigner puede apagar la estela
                prev.clear()

            for px, py, v in puntos_vista:
                se_mueve = v > UMBRAL_VISTA
                cv2.circle(vista, (px, py), 10 if se_mueve else 5, (0, 255, 0) if se_mueve else (200, 200, 200), -1)
            estado = "persona detectada" if puntos_vista else "sin persona"
            cv2.putText(vista, f"{estado} | {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Shadow Dance - vista previa (q para salir)", vista)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    main(int(arg) if arg.isdigit() else arg)
