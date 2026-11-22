"""
pose_osc.py  -  Shadow Dance
Cámara (o video) -> MediaPipe Pose -> posición y velocidad de cada articulación -> OSC a TouchDesigner.

Ejemplos:
    python src/pose_osc.py --camara 1                    # cámara USB número 1
    python src/pose_osc.py --camara 1 --ancho 640 --alto 480
    python src/pose_osc.py --video grabacion.mp4         # video grabado (para pruebas repetibles)
    python src/pose_osc.py --camara 1 --log metricas.csv # guarda FPS y detección cuadro a cuadro

Salir: tecla q en la ventana de vista previa.
La ventana muestra la estela de colores sobre la cámara (se desactiva con --sin-estela).
Al salir imprime un resumen (FPS promedio y % de cuadros con persona): sirve como evidencia de evaluación.
"""
import argparse
import csv
import os
import time

import sys

import cv2
import mediapipe as mp
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postura import CLASE_IDX, ClasificadorVivo  # noqa: E402  (clasificador de postura entrenado)

# ============================ CONFIGURACIÓN ============================
MODEL_NAME = "pose_landmarker_full.task"   # modelo preentrenado de Google (se descarga aparte)
OSC_IP, OSC_PORT = "127.0.0.1", 7000       # a dónde se envían los datos (TouchDesigner escucha aquí)
ALPHA = 0.5                                # suavizado: 1 = sin suavizar, menor = más suave pero con más retraso
MIN_VISIBILITY = 0.5                       # por debajo de esta confianza, la articulación no deja estela
UMBRAL_VISTA = 0.15                        # velocidad (anchos de imagen por segundo) a partir de la cual una articulación "se mueve"
DECAY = 0.92                               # fracción de la estela que se conserva en cada cuadro (menor = estela más corta)
GROSOR_ESTELA = 6                          # grosor de la línea de la estela, en píxeles

# Un color por articulación para la estela de la ventana (BGR, el orden que usa OpenCV)
COLORES = {
    "lw": (0, 255, 255),   # amarillo
    "rw": (0, 165, 255),   # naranja
    "le": (0, 255, 0),     # verde
    "re": (255, 255, 0),   # cian
    "la": (255, 0, 255),   # magenta
    "ra": (255, 0, 0),     # azul
    "lk": (0, 0, 255),     # rojo
    "rk": (255, 0, 128),   # violeta
    "ls": (255, 160, 60),  # hombro izquierdo: azul claro
    "rs": (160, 80, 255),  # hombro derecho: rosa
    "lh": (60, 255, 160),  # cadera izquierda: verde lima
    "rh": (100, 200, 255), # cadera derecha: naranja claro
}

# Articulaciones que dejan estela. Número = índice del punto en MediaPipe Pose (hay 33 en total).
# El nombre corto es el que viaja por OSC: lw = left wrist (muñeca izquierda), rw = right wrist, etc.
JOINTS = {
    "lw": 15, "rw": 16,   # muñecas
    "le": 13, "re": 14,   # codos
    "la": 27, "ra": 28,   # tobillos
    "lk": 25, "rk": 26,   # rodillas
    "ls": 11, "rs": 12,   # hombros (para unir hombro-codo-muñeca en una sola cinta)
    "lh": 23, "rh": 24,   # caderas (para unir cadera-rodilla-tobillo)
}

# "Controladores" de video de Windows. dshow suele ir mejor con webcams USB.
BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", 700),
    "msmf": getattr(cv2, "CAP_MSMF", 1400),
    "auto": getattr(cv2, "CAP_ANY", 0),
}
_AQUI = os.path.dirname(os.path.abspath(__file__))


class Estela:
    """
    Dibuja en la ventana de vista previa la estela de colores de cada articulación que se mueve.
    Funciona con un "lienzo" del mismo tamaño que la imagen:
      1. En cada cuadro se atenúa todo el lienzo (por DECAY): así lo viejo se va apagando.
      2. Por cada articulación más rápida que UMBRAL_VISTA se traza una línea desde su posición
         anterior hasta la nueva: la estela queda continua, sin "perlas" sueltas.
      3. El lienzo se suma a la imagen de la cámara.
    """

    def __init__(self, ancho, alto):
        """Crea el lienzo negro del tamano de la imagen."""
        self.lienzo = np.zeros((alto, ancho, 3), np.float32)  # decimales: así el apagado no se queda trabado
        self.anterior = {}                                    # última posición dibujada de cada articulación

    def actualizar(self, puntos):
        """puntos: lista de (nombre, x_pixel, y_pixel, velocidad). Lista vacía = nadie en cuadro."""
        self.lienzo *= DECAY
        self.lienzo[self.lienzo < 3] = 0                      # lo casi invisible pasa a negro del todo
        for nombre, px, py, vel in puntos:
            if vel > UMBRAL_VISTA:
                color = COLORES[nombre]
                if nombre in self.anterior:
                    cv2.line(self.lienzo, self.anterior[nombre], (px, py), color, GROSOR_ESTELA)
                else:
                    cv2.circle(self.lienzo, (px, py), GROSOR_ESTELA // 2, color, -1)
                self.anterior[nombre] = (px, py)
            else:
                self.anterior.pop(nombre, None)               # al detenerse, se corta la línea
        if not puntos:
            self.anterior.clear()

    def sobre(self, imagen):
        """Devuelve la imagen de la cámara con la estela sumada encima."""
        return cv2.add(imagen, np.clip(self.lienzo, 0, 255).astype(np.uint8))


def buscar_modelo():
    """Busca el archivo del modelo junto a este script y, si no está, en la carpeta actual."""
    for ruta in (os.path.join(_AQUI, MODEL_NAME), MODEL_NAME):
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        f"No encuentro {MODEL_NAME}. Descárgalo (ver instrucciones) y ponlo en la carpeta src/.")


def buscar_modelo_postura(ruta):
    """Busca modelos\\postura.joblib (la carpeta modelos esta junto a src)."""
    for candidato in (ruta, os.path.join("modelos", "postura.joblib"),
                      os.path.join(_AQUI, "..", "modelos", "postura.joblib")):
        if candidato and os.path.isfile(candidato):
            return candidato
    return None


def abrir_fuente(args):
    """Abre la cámara USB (o el video) y configura resolución y velocidad."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        descripcion = f"video {args.video}"
    else:
        cap = cv2.VideoCapture(args.camara, BACKENDS[args.backend])
        # MJPG = la cámara comprime cada imagen. Las webcams USB 2.0 suelen necesitarlo
        # para dar 30 cuadros por segundo en resolución alta (sin comprimir no alcanza el ancho de banda).
        # Con --sin-mjpg no se pide ese formato (la PS3 Eye, por ejemplo, solo entrega video sin comprimir).
        if not args.sin_mjpg:
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
    """Programa de siempre: abre la fuente, detecta la pose, clasifica la postura y envia todo por OSC."""
    parser = argparse.ArgumentParser(description="Shadow Dance: cámara -> pose -> OSC")
    parser.add_argument("--camara", type=int, default=0, help="número de la cámara (usa listar_camaras.py)")
    parser.add_argument("--video", default=None, help="ruta de un video; si se da, se ignora la cámara")
    parser.add_argument("--ancho", type=int, default=1280, help="ancho pedido a la cámara")
    parser.add_argument("--alto", type=int, default=720, help="alto pedido a la cámara")
    parser.add_argument("--fps", type=int, default=30, help="cuadros por segundo pedidos a la cámara")
    parser.add_argument("--backend", choices=list(BACKENDS), default="dshow", help="controlador de video")
    parser.add_argument("--sin-mjpg", action="store_true", help="no pedir formato MJPG (para cámaras como la PS3 Eye)")
    parser.add_argument("--sin-espejo", action="store_true", help="desactiva el efecto espejo")
    parser.add_argument("--sin-estela", action="store_true", help="no dibuja la estela en la ventana de vista previa")
    parser.add_argument("--log", default=None, help="archivo CSV donde guardar FPS y detección por cuadro")
    parser.add_argument("--modelo-postura", default=None, help="ruta de postura.joblib (por defecto modelos\\postura.joblib)")
    parser.add_argument("--sin-postura", action="store_true", help="no clasifica la postura (solo estela)")
    parser.add_argument("--ventana-postura", type=int, default=5, help="votos que se promedian (más = más estable, más lento)")
    parser.add_argument("--cada-postura", type=int, default=2, help="clasificar 1 de cada N cuadros")
    args = parser.parse_args()
    espejo = not args.sin_espejo

    # --- 0. Clasificador de postura (entrenado por nosotros; es opcional) ---
    clasificador = None
    if not args.sin_postura:
        ruta_post = buscar_modelo_postura(args.modelo_postura)
        if ruta_post:
            clasificador = ClasificadorVivo(ruta_post, args.ventana_postura, args.cada_postura)
            print(f"Clasificador de postura cargado: {ruta_post} ({clasificador.tipo})")
        else:
            print("No encuentro modelos\\postura.joblib: sigo sin clasificar la postura "
                  "(entrena con: python src\\entrenar_clasificador.py --guardar)")

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
    estela = None             # se crea con el primer cuadro (necesita su tamaño)
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
            postura = None

            # --- 5. Calcular y enviar los datos por OSC ---
            if resultado.pose_landmarks:
                client.send_message("/detected", 1.0)
                puntos = resultado.pose_landmarks[0]   # los 33 puntos de la primera persona
                if clasificador:
                    # El clasificador usa las coordenadas SIN espejo (con las que se entrenó).
                    brazos, piernas, postura = clasificador.actualizar([(p.x, p.y) for p in puntos])
                    client.send_message("/posture", float(CLASE_IDX[postura]))   # 0 ninguna, 1 brazos, 2 piernas, 3 ambas
                    client.send_message("/arms_open", float(brazos))
                    client.send_message("/legs_open", float(piernas))
                for nombre, idx in JOINTS.items():
                    p = puntos[idx]
                    x = 1.0 - p.x if espejo else p.x   # espejo: lo que sale por la derecha de la cámara aparece a la izquierda
                    x, y, vel = suavizar_y_velocidad(prev, nombre, x, p.y, dt, ALPHA)
                    if p.visibility < MIN_VISIBILITY:
                        vel = 0.0                      # punto dudoso: no dejará estela
                    client.send_message(f"/{nombre}_x", float(x))
                    client.send_message(f"/{nombre}_y", float(y))
                    client.send_message(f"/{nombre}_s", float(vel))
                    puntos_vista.append((nombre, int(x * ancho), int(y * alto), vel))
            else:
                client.send_message("/detected", 0.0)  # sin persona: TouchDesigner puede apagar la estela
                prev.clear()
                if clasificador:
                    clasificador.reiniciar()
                    for canal in ("/posture", "/arms_open", "/legs_open"):
                        client.send_message(canal, 0.0)

            # --- 6. Métricas para la evaluación ---
            total += 1
            con_persona += 1 if puntos_vista else 0
            if writer:
                writer.writerow([f"{t - t0:.3f}", f"{fps:.1f}", 1 if puntos_vista else 0])

            # --- 7. Ventana de vista previa: estela + puntos + texto ---
            if not args.sin_estela:
                if estela is None:
                    estela = Estela(ancho, alto)
                estela.actualizar(puntos_vista)
                vista = estela.sobre(vista)
            for _, px, py, v in puntos_vista:
                se_mueve = v > UMBRAL_VISTA
                cv2.circle(vista, (px, py), 10 if se_mueve else 5,
                           (0, 255, 0) if se_mueve else (200, 200, 200), -1)
            estado = "persona detectada" if puntos_vista else "sin persona"
            cv2.putText(vista, f"{estado} | {fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if postura:
                cv2.putText(vista, f"postura: {postura.upper()}", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
        if clasificador and clasificador.n_pred:
            print(f"Clasificador de postura: {clasificador.ms_promedio():.1f} ms por predicción "
                  f"({clasificador.n_pred} predicciones)")
        if args.log:
            print(f"Registro guardado en {args.log}")


if __name__ == "__main__":
    main()
