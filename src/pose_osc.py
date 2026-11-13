"""
pose_osc.py  -  Shadow Dance
Cámara (o video) -> MediaPipe Pose -> posición y velocidad de cada articulación -> OSC a TouchDesigner.

Ejemplos:
    python src/pose_osc.py --camara 1                    # cámara USB número 1
    python src/pose_osc.py --camara 1 --ancho 640 --alto 480
    python src/pose_osc.py --video grabacion.mp4         # video grabado (para pruebas repetibles)
    python src/pose_osc.py --camara 1 --log metricas.csv # guarda FPS y detección cuadro a cuadro

Salir: tecla q en la ventana de vista previa.
Al salir imprime un resumen (FPS promedio y % de cuadros con persona): sirve como evidencia de evaluación.
"""
import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

# ============================ CONFIGURACIÓN ============================
MODEL_NAME = "pose_landmarker_full.task"   # modelo preentrenado de Google (se descarga aparte)
OSC_IP, OSC_PORT = "127.0.0.1", 7000       # a dónde se envían los datos (TouchDesigner escucha aquí)
ALPHA = 0.5                                # suavizado: 1 = sin suavizar, menor = más suave pero con más retraso
MIN_VISIBILITY = 0.5                       # por debajo de esta confianza, la articulación no deja estela
UMBRAL_VISTA = 0.15                        # solo para la ventana: velocidad a partir de la cual el punto se pinta verde

# Articulaciones que dejan estela. Número = índice del punto en MediaPipe Pose (hay 33 en total).
# El nombre corto es el que viaja por OSC: lw = left wrist (muñeca izquierda), rw = right wrist, etc.
JOINTS = {
    "lw": 15, "rw": 16,   # muñecas
    "le": 13, "re": 14,   # codos
    "la": 27, "ra": 28,   # tobillos
    "lk": 25, "rk": 26,   # rodillas
}

# "Controladores" de video de Windows. dshow suele ir mejor con webcams USB.
BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", 700),
    "msmf": getattr(cv2, "CAP_MSMF", 1400),
    "auto": getattr(cv2, "CAP_ANY", 0),
}
_AQUI = os.path.dirname(os.path.abspath(__file__))


def buscar_modelo():
    """Busca el archivo del modelo junto a este script y, si no está, en la carpeta actual."""
    for ruta in (os.path.join(_AQUI, MODEL_NAME), MODEL_NAME):
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        f"No encuentro {MODEL_NAME}. Descárgalo (ver instrucciones) y ponlo en la carpeta src/.")


def abrir_fuente(args):
    """Abre la cámara USB (o el video) y configura resolución y velocidad."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        descripcion = f"video {args.video}"
    else:
        cap = cv2.VideoCapture(args.camara, BACKENDS[args.backend])
        # MJPG = la cámara comprime cada imagen. Las webcams USB 2.0 suelen necesitarlo
        # para dar 30 cuadros por segundo en resolución alta (sin comprimir no alcanza el ancho de banda).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # guarda pocos cuadros en cola = menos retraso (si la cámara lo permite)
        descripcion = f"cámara {args.camara} ({args.backend})"

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir {descripcion}. Revisa que no la use otro programa (Zoom, Teams, el navegador) "
            "y prueba otro número con listar_camaras.py.")
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Fuente abierta: {descripcion} - {ancho}x{alto} a {fps:.0f} FPS (valores que reporta la cámara)")
    return cap


def suavizar_y_velocidad(prev, nombre, x, y, dt, alpha):
    """
    Suaviza la posición mezclando la nueva con la anterior (media móvil exponencial, EMA):
        nueva_suavizada = alpha * medida + (1 - alpha) * anterior
    Esto quita el "temblor" del detector. Luego calcula la velocidad:
        velocidad = distancia recorrida / tiempo entre cuadros
    (en "anchos de imagen por segundo": 1.0 = cruzó todo el ancho de la imagen en un segundo).
    """
    if nombre in prev and dt:
        px, py = prev[nombre]
        x = alpha * x + (1 - alpha) * px
        y = alpha * y + (1 - alpha) * py
        vel = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 / dt
    else:
        vel = 0.0  # primer cuadro: aún no hay con qué comparar
    prev[nombre] = (x, y)
    return x, y, vel


def main():
    parser = argparse.ArgumentParser(description="Shadow Dance: cámara -> pose -> OSC")
    parser.add_argument("--camara", type=int, default=0, help="número de la cámara (usa listar_camaras.py)")
    parser.add_argument("--video", default=None, help="ruta de un video; si se da, se ignora la cámara")
    parser.add_argument("--ancho", type=int, default=1280, help="ancho pedido a la cámara")
    parser.add_argument("--alto", type=int, default=720, help="alto pedido a la cámara")
    parser.add_argument("--fps", type=int, default=30, help="cuadros por segundo pedidos a la cámara")
    parser.add_argument("--backend", choices=list(BACKENDS), default="dshow", help="controlador de video")
    parser.add_argument("--sin-espejo", action="store_true", help="desactiva el efecto espejo")
    parser.add_argument("--log", default=None, help="archivo CSV donde guardar FPS y detección por cuadro")
    args = parser.parse_args()
    espejo = not args.sin_espejo

    # --- 1. Preparar el modelo de pose (preentrenado) ---
    opciones = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=buscar_modelo()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,  # procesa cuadro por cuadro, en orden
        num_poses=1,                                      # una sola persona
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # --- 2. Preparar el envío OSC y la fuente de video ---
    client = SimpleUDPClient(OSC_IP, OSC_PORT)
    cap = abrir_fuente(args)

    archivo_log = writer = None
    if args.log:
        archivo_log = open(args.log, "w", newline="", encoding="utf-8")
        writer = csv.writer(archivo_log)
        writer.writerow(["tiempo_s", "fps", "persona_detectada"])

    prev = {}                 # última posición suavizada de cada articulación
    t_prev = None
    fps = 0.0
    total, con_persona = 0, 0
    t0 = time.time()

    with mp.tasks.vision.PoseLandmarker.create_from_options(opciones) as landmarker:
        while True:
            # --- 3. Leer un cuadro ---
            ok, frame = cap.read()
            if not ok:
                print("La fuente dejó de entregar imágenes (fin del video o cámara desconectada).")
                break
            t = time.time()
            dt = (t - t_prev) if t_prev else None
            t_prev = t
            if dt:
                fps = (1 / dt) if fps == 0 else 0.9 * fps + 0.1 * (1 / dt)  # FPS suavizado para mostrarlo

            # --- 4. Detectar la pose (MediaPipe espera RGB; OpenCV entrega BGR) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultado = landmarker.detect_for_video(mp_image, int((t - t0) * 1000))

            vista = cv2.flip(frame, 1) if espejo else frame.copy()
            alto, ancho = vista.shape[:2]
            puntos_vista = []

            # --- 5. Calcular y enviar los datos por OSC ---
            if resultado.pose_landmarks:
                client.send_message("/detected", 1.0)
                puntos = resultado.pose_landmarks[0]   # los 33 puntos de la primera persona
                for nombre, idx in JOINTS.items():
                    p = puntos[idx]
                    x = 1.0 - p.x if espejo else p.x   # espejo: lo que sale por la derecha de la cámara aparece a la izquierda
                    x, y, vel = suavizar_y_velocidad(prev, nombre, x, p.y, dt, ALPHA)
                    if p.visibility < MIN_VISIBILITY:
                        vel = 0.0                      # punto dudoso: no dejará estela
                    client.send_message(f"/{nombre}_x", float(x))
                    client.send_message(f"/{nombre}_y", float(y))
                    client.send_message(f"/{nombre}_s", float(vel))
                    puntos_vista.append((int(x * ancho), int(y * alto), vel))
            else:
                client.send_message("/detected", 0.0)  # sin persona: TouchDesigner puede apagar la estela
                prev.clear()

            # --- 6. Métricas para la evaluación ---
            total += 1
            con_persona += 1 if puntos_vista else 0
            if writer:
                writer.writerow([f"{t - t0:.3f}", f"{fps:.1f}", 1 if puntos_vista else 0])

            # --- 7. Ventana de vista previa ---
            for px, py, v in puntos_vista:
                se_mueve = v > UMBRAL_VISTA
                cv2.circle(vista, (px, py), 10 if se_mueve else 5,
                           (0, 255, 0) if se_mueve else (200, 200, 200), -1)
            estado = "persona detectada" if puntos_vista else "sin persona"
            cv2.putText(vista, f"{estado} | {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Shadow Dance - vista previa (q para salir)", vista)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    if archivo_log:
        archivo_log.close()

    # --- 8. Resumen final ---
    duracion = time.time() - t0
    if total:
        print("\n===== RESUMEN =====")
        print(f"Cuadros procesados: {total} en {duracion:.1f} s  ->  {total / duracion:.1f} FPS promedio")
        print(f"Cuadros con persona detectada: {con_persona} ({100 * con_persona / total:.1f} %)")
        if args.log:
            print(f"Registro guardado en {args.log}")


if __name__ == "__main__":
    main()
