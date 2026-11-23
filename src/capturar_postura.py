"""
capturar_postura.py  -  Shadow Dance
====================================
Graba datos PROPIOS con la camara en vivo, ya etiquetados por el protocolo (sin etiquetar a mano):
tu pulsas la tecla de la postura, hay una cuenta atras para que te pongas en posicion, y durante
unos segundos se guardan los 33 puntos del cuerpo (solo numeros, no imagenes) con esa etiqueta.

Teclas (en la ventana):
    1 = brazos abiertos (piernas juntas)      3 = brazos Y piernas abiertos
    2 = piernas abiertas (brazos abajo)       4 = ninguna (de pie, quieto, brazos y piernas cerrados)
    espacio = cancelar la toma en curso       q = salir

Cada toma = un "grupo" distinto: asi, al evaluar, las tomas de prueba nunca se mezclan con las de entrenamiento.

Uso (desde la raiz del proyecto):
    python src\\capturar_postura.py --camara 1 --sin-mjpg --ancho 640 --alto 480 --sesion adulto1
    (para otra persona o dia: cambia --sesion adulto2, o usa otro --persona)

Recomendado por sesion: 2 tomas de cada postura (8 tomas), variando un poco la distancia a la camara y
moviendo el cuerpo despacio entre cuadros. Cuerpo COMPLETO en cuadro (de la cabeza a los pies).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extraer_puntos import CAMPOS, N_PUNTOS  # noqa: E402  (mismas columnas que keypoints.csv)

TECLAS = {ord("1"): "brazos", ord("2"): "piernas", ord("3"): "ambas", ord("4"): "ninguna"}
TEXTO = {"brazos": "BRAZOS abiertos", "piernas": "PIERNAS abiertas",
         "ambas": "BRAZOS + PIERNAS abiertos", "ninguna": "NINGUNA (quieto)"}
PUNTOS_CUERPO = (11, 12, 23, 24, 25, 26, 27, 28)   # hombros, caderas, rodillas, tobillos
MODELO = "pose_landmarker_full.task"
_AQUI = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- funciones puras ---
def cuerpo_completo(puntos, minimo=0.5):
    """True si hombros, caderas, rodillas y tobillos se ven con confianza media >= minimo."""
    vis = [puntos[i][3] for i in PUNTOS_CUERPO]
    return sum(vis) / len(vis) >= minimo


def columnas():
    """Nombres de las columnas del CSV (las mismas que data/keypoints.csv): datos de la toma + 33 puntos x 4 valores."""
    base = ["archivo", "numero", "clase", "clase_original", "grupo", "persona", "hash", "detectada"]
    return base + [f"p{i}_{c}" for i in range(N_PUNTOS) for c in CAMPOS]


def fila_csv(sesion, persona, clase, toma, k, puntos):
    """Una fila con el mismo formato que data/keypoints.csv. k = numero de cuadro guardado en la toma."""
    nombre = f"{sesion}_t{toma}_{clase}_{k:04d}"
    grupo = f"{persona}{sesion}{toma}"        # una toma = un grupo
    base = [nombre, "", clase, clase, grupo, persona, nombre, 1]
    return base + [v for p in puntos for v in p]


class Toma:
    """
    Maquina de estados de una toma: espera -> cuenta atras -> grabando -> espera.
    Recibe el tiempo desde afuera, asi se prueba sin camara.
    """
    def __init__(self, cuenta=3.0, duracion=12.0):
        self.cuenta, self.duracion = cuenta, duracion
        self.clase, self.inicio, self.numero = None, None, 0

    def iniciar(self, clase, t):
        self.clase, self.inicio = clase, t
        self.numero += 1

    def cancelar(self):
        self.clase = self.inicio = None

    def estado(self, t):
        """('espera'|'cuenta'|'grabando', segundos_restantes)."""
        if self.clase is None:
            return "espera", 0.0
        dt = t - self.inicio
        if dt < self.cuenta:
            return "cuenta", self.cuenta - dt
        if dt < self.cuenta + self.duracion:
            return "grabando", self.cuenta + self.duracion - dt
        self.clase = self.inicio = None
        return "espera", 0.0


# --------------------------------------------------------------------------------------- camara ---
def buscar_modelo():
    """Ruta de pose_landmarker_full.task (junto a este script o en la carpeta actual); si no esta, termina."""
    for ruta in (os.path.join(_AQUI, MODELO), MODELO):
        if os.path.exists(ruta):
            return ruta
    sys.exit(f"No encuentro {MODELO} en src\\")


def main():
    # --- 1. opciones de la linea de comandos ---
    ap = argparse.ArgumentParser(description="Captura de posturas etiquetadas con la camara en vivo")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--sin-mjpg", action="store_true")
    ap.add_argument("--backend", choices=["dshow", "msmf", "auto"], default="dshow")
    ap.add_argument("--salida", default=os.path.join("data", "captura.csv"), help="CSV (se agrega al final si existe)")
    ap.add_argument("--sesion", default="s1", help="nombre de la sesion, ej. adulto1 (sin espacios)")
    ap.add_argument("--persona", default="C", help="letra de la persona (por defecto C = adulto en vivo)")
    ap.add_argument("--cada", type=int, default=3, help="guardar 1 de cada N cuadros (por defecto 3)")
    ap.add_argument("--cuenta", type=float, default=3.0, help="segundos de cuenta atras")
    ap.add_argument("--duracion", type=float, default=12.0, help="segundos grabando por toma")
    ap.add_argument("--escala", type=float, default=2.0, help="tamano de la ventana (2 = el doble; 1 = original)")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    # --- 2. abrir la camara (DirectShow en Windows) ---
    back = {"dshow": getattr(cv2, "CAP_DSHOW", 700), "msmf": getattr(cv2, "CAP_MSMF", 1400),
            "auto": getattr(cv2, "CAP_ANY", 0)}[args.backend]
    cap = cv2.VideoCapture(args.camara, back)
    if not args.sin_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
    if not cap.isOpened():
        sys.exit("No se pudo abrir la camara (cierra otros programas que la usen).")

    # --- 3. abrir el CSV de salida (si ya existe, se agregan filas al final) ---
    nuevo = not os.path.exists(args.salida)
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    f = open(args.salida, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if nuevo:
        w.writerow(columnas())

    # --- 4. cargar MediaPipe Pose (modo VIDEO) y preparar la maquina de estados de la toma ---
    opciones = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=buscar_modelo()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)

    toma = Toma(args.cuenta, args.duracion)
    guardadas, descartadas, k, n_cuadro = {}, 0, 0, 0
    t0 = time.time()
    print("Teclas: 1 brazos | 2 piernas | 3 ambas | 4 ninguna | espacio cancela | q sale")

    # --- 5. bucle principal: leer cuadro -> detectar -> (si se graba) guardar -> dibujar -> teclas ---
    with mp.tasks.vision.PoseLandmarker.create_from_options(opciones) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time()
            # detectar los 33 puntos y comprobar que el cuerpo entero esta en cuadro
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((t - t0) * 1000))
            puntos = None
            if res.pose_landmarks:
                puntos = [(p.x, p.y, p.z, p.visibility) for p in res.pose_landmarks[0]]
            completo = puntos is not None and cuerpo_completo(puntos)

            # guardar 1 de cada N cuadros mientras la toma esta grabando (solo si se ve el cuerpo completo)
            estado, resto = toma.estado(t)
            if estado == "grabando":
                n_cuadro += 1
                if n_cuadro % args.cada == 0:
                    if completo:
                        k += 1
                        w.writerow(fila_csv(args.sesion, args.persona, toma.clase, toma.numero, k, puntos))
                        guardadas[toma.clase] = guardadas.get(toma.clase, 0) + 1
                    else:
                        descartadas += 1
            elif estado == "espera":
                f.flush()

            # dibujar la ventana: imagen en espejo, puntos (verde = cuerpo completo, rojo = incompleto) y mensajes
            vista = cv2.flip(frame, 1)
            if args.escala != 1.0:                 # solo agranda lo que SE VE; los puntos guardados no cambian
                vista = cv2.resize(vista, None, fx=args.escala, fy=args.escala, interpolation=cv2.INTER_LINEAR)
            h, ancho = vista.shape[:2]
            if puntos:
                for i in PUNTOS_CUERPO + (13, 14, 15, 16):
                    cv2.circle(vista, (int((1 - puntos[i][0]) * ancho), int(puntos[i][1] * h)), 5,
                               (0, 255, 0) if completo else (0, 0, 255), -1)
            if estado == "cuenta":
                msg, col = f"{TEXTO[toma.clase]} - listo en {resto:.0f}...", (0, 255, 255)
            elif estado == "grabando":
                msg, col = f"GRABANDO {TEXTO[toma.clase]}  ({resto:.0f} s)", (0, 0, 255)
            else:
                msg, col = "1 brazos | 2 piernas | 3 ambas | 4 ninguna", (255, 255, 255)
            cv2.putText(vista, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            if not completo:
                cv2.putText(vista, "cuerpo incompleto: aleja la camara", (10, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            resumen = " ".join(f"{c}:{guardadas.get(c, 0)}" for c in TEXTO)
            cv2.putText(vista, resumen, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Shadow Dance - captura (q sale)", vista)

            # teclas: q sale, espacio cancela la toma, 1-4 inician una toma nueva
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord(" "):
                toma.cancelar()
            elif tecla in TECLAS and estado == "espera":
                toma.iniciar(TECLAS[tecla], t)
                k = 0

    # --- 6. cerrar todo y mostrar el resumen ---
    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nGuardado en {args.salida}: {guardadas}  (cuadros descartados por cuerpo incompleto: {descartadas})")


if __name__ == "__main__":
    main()
