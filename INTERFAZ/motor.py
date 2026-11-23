"""
motor.py  -  Interfaz de Shadow Dance
=====================================
El "motor" que usa la interfaz: lee la camara (o un video), detecta el cuerpo con MediaPipe, clasifica la
postura y manda los datos por OSC a TouchDesigner, igual que src\\pose_osc.py. REUTILIZA ese codigo
(importa sus funciones y constantes) y no modifica nada de src\\.

Lo que agrega para la interfaz:
  - corre en un hilo y deja disponible el ultimo cuadro (ya con el esqueleto dibujado) y las cifras en vivo
  - se puede apagar y prender la deteccion sin cerrar la camara
  - modo video: reproduce un clip a su velocidad real (con repeticion opcional) y puede mandarlo por OSC
  - opcion CLAHE (ecualizacion de histograma) antes de detectar, para poca luz
"""
import os
import sys
import threading
import time
from types import SimpleNamespace

import cv2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import mediapipe as mp                       # noqa: E402
from pythonosc.udp_client import SimpleUDPClient  # noqa: E402

import pose_osc as P                         # noqa: E402  (el programa de siempre: se usa, no se cambia)
from evaluar_sistema import ecualizar_clahe  # noqa: E402
from postura import ClasificadorVivo, CLASE_IDX  # noqa: E402

# ---------------------------------------------------------------------------------- constantes ---
ANCHO_MAX = 640                              # los videos grandes se reducen a este ancho (como la camara en vivo)
ARTICULACIONES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

# Solo se dibujan las EXTREMIDADES (sin torso). Colores en BGR (el orden de OpenCV).
BRAZOS = [(11, 13), (13, 15), (12, 14), (14, 16)]
PIERNAS = [(23, 25), (25, 27), (24, 26), (26, 28)]
PUNTOS_DIBUJADOS = (13, 14, 15, 16, 25, 26, 27, 28)              # codos, munecas, rodillas y tobillos (sin hombros ni caderas)
C_BRAZOS, C_BRAZOS_ABIERTOS = (255, 217, 0), (13, 191, 255)      # cian / dorado
C_PIERNAS, C_PIERNAS_ABIERTAS = (191, 25, 255), (0, 89, 255)     # magenta / naranja


def buscar_modelo_postura():
    """Ruta de modelos\\postura.joblib (el clasificador entrenado) o None si todavia no se ha entrenado."""
    ruta = os.path.join(RAIZ, "modelos", "postura.joblib")
    return ruta if os.path.isfile(ruta) else None


def _ruta_modelo(cfg):
    """Modelo de pose. Por defecto el 'full'; si config.json pone "modelo": "lite" y existe
    src\\pose_landmarker_lite.task, usa ese (mas rapido, un poco menos preciso)."""
    if str(cfg.get("modelo", "full")).lower() == "lite":
        lite = os.path.join(RAIZ, "src", "pose_landmarker_lite.task")
        if os.path.isfile(lite):
            return lite
    return P.buscar_modelo()


class Motor:
    """Un hilo que lee la fuente, detecta, clasifica y envia OSC; la interfaz lo controla con sus atributos."""

    def __init__(self, cfg, fuente=None, osc=True, detectar=True, clahe=False, bucle=True):
        """
        cfg:    diccionario de config.json (camara, ancho, alto, sin_mjpg, backend)
        fuente: ruta de un video; None = camara
        """
        self.cfg, self.fuente = cfg, fuente
        self.es_video = fuente is not None
        self.osc, self.detectar, self.clahe, self.bucle = osc, detectar, clahe, bucle
        self.pausado = False
        self.reiniciar_video = False
        self._parar = threading.Event()
        self._cond = threading.Condition()
        self._jpeg, self._n = None, 0
        self.error = None
        self.activo = False
        self.terminado = False
        self.clasificador_cargado = False
        self.stats = {"fps": 0.0, "ms_cuadro": 0.0, "ms_clasificador": 0.0, "detectado": False,
                      "visibilidad": 0.0, "brazos": None, "piernas": None, "clase": None, "cuadros": 0,
                      "con_persona": 0}
        self._hilo = threading.Thread(target=self._correr, daemon=True)

    # ------------------------------------------------------------------ control desde la interfaz ---
    def iniciar(self):
        """Arranca el hilo del motor."""
        self._hilo.start()

    def detener(self):
        """Pide al hilo que pare y espera a que termine."""
        self._parar.set()
        self._hilo.join(timeout=4)
        with self._cond:
            self._cond.notify_all()

    def esperar_cuadro(self, ultimo, timeout=1.0):
        """Devuelve (jpeg, numero) cuando hay un cuadro mas nuevo que 'ultimo'; (None, ultimo) si se acabo el tiempo."""
        with self._cond:
            self._cond.wait_for(
                lambda: (self._n != ultimo and self._jpeg is not None) or self.terminado or self._parar.is_set(),
                timeout)
            if self._n == ultimo or self._jpeg is None:
                return None, ultimo
            return self._jpeg, self._n

    def estado(self):
        """Cifras en vivo (FPS, deteccion, postura, banderas) que la pagina consulta cada ~0.3 s."""
        s = dict(self.stats)
        s.update(activo=self.activo, modo="video" if self.es_video else "camara", pausado=self.pausado,
                 detectar=self.detectar, clahe=self.clahe, osc=self.osc, bucle=self.bucle, error=self.error,
                 terminado=self.terminado, clasificador=self.clasificador_cargado,
                 fuente=os.path.basename(self.fuente) if self.fuente else None)
        return s

    # ------------------------------------------------------------------------------------ interno ---
    def _publicar(self, img):
        """Guarda el cuadro como JPEG y avisa a quien este esperando el video (/video)."""
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._cond:
                self._jpeg, self._n = buf.tobytes(), self._n + 1
                self._cond.notify_all()

    def _dibujar(self, img, puntos, espejo, brazos_ab, piernas_ab):
        """Dibuja solo brazos y piernas sobre la imagen; las lineas cambian de color si estan 'abiertas'."""
        h, w = img.shape[:2]

        def pt(i):                       # landmark normalizado -> pixel (con espejo si es camara)
            x = 1.0 - puntos[i].x if espejo else puntos[i].x
            return int(x * w), int(puntos[i].y * h)

        for lista, color in ((BRAZOS, C_BRAZOS_ABIERTOS if brazos_ab else C_BRAZOS),
                             (PIERNAS, C_PIERNAS_ABIERTAS if piernas_ab else C_PIERNAS)):
            for a, b in lista:
                if puntos[a].visibility > 0.3 and puntos[b].visibility > 0.3:
                    cv2.line(img, pt(a), pt(b), color, 3, cv2.LINE_AA)
        for i in PUNTOS_DIBUJADOS:
            if puntos[i].visibility > 0.3:
                cv2.circle(img, pt(i), 5, (255, 255, 255), -1, cv2.LINE_AA)

    def _enviar_cero(self, cliente):
        """Avisa a TouchDesigner que no hay nadie: apaga la estela y la postura."""
        if cliente:
            for canal in ("/detected", "/posture", "/arms_open", "/legs_open"):
                cliente.send_message(canal, 0.0)

    def _correr(self):
        """Bucle principal del hilo: preparar -> (leer, detectar, clasificar, enviar, mostrar) en bucle -> limpiar."""
        cfg = self.cfg

        # --- 1. abrir la camara o el video ---
        try:
            args = SimpleNamespace(video=self.fuente, camara=int(cfg.get("camara", 0)),
                                   backend=cfg.get("backend", "dshow"), sin_mjpg=bool(cfg.get("sin_mjpg", True)),
                                   ancho=int(cfg.get("ancho", 640)), alto=int(cfg.get("alto", 480)), fps=30)
            cap = P.abrir_fuente(args)
        except Exception as e:                        # camara ocupada, indice equivocado, video que no abre...
            self.error = str(e)
            self.terminado = True
            return
        # --- 2. cargar MediaPipe Pose (VIDEO mode: sigue a la persona entre cuadros) ---
        try:
            opciones = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=_ruta_modelo(cfg)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            marcador = mp.tasks.vision.PoseLandmarker.create_from_options(opciones)
        except Exception as e:
            self.error = f"No pude cargar el modelo de pose: {e}"
            self.terminado = True
            cap.release()
            return

        # --- 3. clasificador de postura, cliente OSC y variables del bucle ---
        ruta_post = buscar_modelo_postura()
        clasif = ClasificadorVivo(ruta_post, 5, 3) if ruta_post else None
        self.clasificador_cargado = clasif is not None
        cliente = SimpleUDPClient(P.OSC_IP, P.OSC_PORT)
        espejo = not self.es_video            # la camara se ve como espejo; los videos se ven tal cual se grabaron
        fps_video = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.es_video else 30.0
        prev, t_prev, t0 = {}, None, time.time()
        fps_ema, ms_ema, ms_clasif_ema = 0.0, 0.0, 0.0
        # modo adaptativo: si el PC no alcanza a detectar en cada cuadro, detecta 1 de cada 2 y reutiliza
        # la ultima pose en los cuadros intermedios (el video y el dibujo siguen a toda velocidad)
        ult_pts, det_cada, n_cuadro = None, 1, 0
        ms_det_ema, ms_read_ema = 0.0, 0.0
        enviado_cero = False
        self.activo = True

        try:
            while not self._parar.is_set():
                # pausa / reinicio pedidos desde la interfaz
                if self.pausado:
                    time.sleep(0.05)
                    t_prev = None
                    continue
                if self.reiniciar_video and self.es_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ult_pts, n_cuadro = None, 0
                    prev.clear()
                    clasif and clasif.reiniciar()
                    self.reiniciar_video = False
                # --- leer un cuadro (en video, al terminar vuelve al inicio si "Repetir" esta activo) ---
                inicio = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if self.es_video and self.bucle:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ult_pts, n_cuadro = None, 0
                        prev.clear()
                        clasif and clasif.reiniciar()
                        continue
                    self.terminado = True
                    self.error = None if self.es_video else "La camara dejo de entregar imagenes."
                    break

                # --- preparar el cuadro: reducir, medir tiempo entre cuadros, CLAHE opcional, vista espejo ---
                t_proc = time.perf_counter()              # desde aqui se mide el trabajo propio (sin esperar a la camara)
                if frame.shape[1] > ANCHO_MAX:
                    alto = int(round(frame.shape[0] * ANCHO_MAX / frame.shape[1]))
                    frame = cv2.resize(frame, (ANCHO_MAX, alto), interpolation=cv2.INTER_AREA)
                t = time.time()
                if self.es_video:                          # decodificar un video pesado (4K) tambien cuesta tiempo
                    ms_read_ema = (t_proc - inicio) * 1000 if ms_read_ema == 0 else 0.9 * ms_read_ema + 0.1 * (t_proc - inicio) * 1000

                if self.clahe:
                    frame = ecualizar_clahe(frame)
                vista = cv2.flip(frame, 1) if espejo else frame.copy()
                persona, brazos_ab, piernas_ab = False, False, False

                # --- detectar: MediaPipe -> clasificador -> OSC -> dibujo ---
                if self.detectar:
                    enviado_cero = False
                    nueva = (n_cuadro % det_cada == 0)       # en este cuadro se corre MediaPipe de verdad?
                    n_cuadro += 1
                    if nueva:
                        dt = (t - t_prev) if t_prev else None
                        t_prev = t
                        t_det = time.perf_counter()
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = marcador.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                        int((t - t0) * 1000))
                        ms_det = (time.perf_counter() - t_det) * 1000
                        ms_det_ema = ms_det if ms_det_ema == 0 else 0.9 * ms_det_ema + 0.1 * ms_det
                        ult_pts = res.pose_landmarks[0] if res.pose_landmarks else None
                        # adaptativo (con histeresis): coste estimado de un cuadro completo vs. lo que dura un cuadro
                        coste, periodo = ms_det_ema + 8.0 + ms_read_ema, 1000.0 / fps_video
                        if det_cada == 1 and coste > 1.1 * periodo:
                            det_cada = 2
                        elif det_cada == 2 and coste < 0.7 * periodo:
                            det_cada = 1
                        self.stats["det_cada"] = det_cada
                    puntos = ult_pts
                    if puntos is not None:
                        persona = True
                        if clasif and not nueva:             # cuadro intermedio: se mantiene la ultima postura
                            brazos_ab, piernas_ab = bool(clasif.estado[0]), bool(clasif.estado[1])
                        if nueva and self.osc:
                            cliente.send_message("/detected", 1.0)
                        if nueva and clasif:                 # postura: 2 preguntas si/no (brazos, piernas) -> clase
                            b, p, clase = clasif.actualizar([(q.x, q.y) for q in puntos])
                            brazos_ab, piernas_ab = bool(b), bool(p)
                            if clasif.n_pred:
                                ms_clasif_ema = clasif.ms_promedio()
                            if self.osc:
                                cliente.send_message("/posture", float(CLASE_IDX[clase]))
                                cliente.send_message("/arms_open", float(b))
                                cliente.send_message("/legs_open", float(p))
                            self.stats["brazos"] = 100.0 * sum(clasif.hist["abre_brazos"]) / max(1, len(clasif.hist["abre_brazos"]))
                            self.stats["piernas"] = 100.0 * sum(clasif.hist["abre_piernas"]) / max(1, len(clasif.hist["abre_piernas"]))
                            self.stats["clase"] = clase
                        if nueva:
                            for nombre, idx in P.JOINTS.items():     # 12 articulaciones: x, y suavizados y velocidad
                                q = puntos[idx]
                                x = 1.0 - q.x if espejo else q.x
                                x, y, vel = P.suavizar_y_velocidad(prev, nombre, x, q.y, dt, P.ALPHA)
                                if q.visibility < P.MIN_VISIBILITY:
                                    vel = 0.0
                                if self.osc:
                                    cliente.send_message(f"/{nombre}_x", float(x))
                                    cliente.send_message(f"/{nombre}_y", float(y))
                                    cliente.send_message(f"/{nombre}_s", float(vel))
                            self.stats["visibilidad"] = 100.0 * sum(puntos[i].visibility for i in ARTICULACIONES) / len(ARTICULACIONES)
                        self._dibujar(vista, puntos, espejo, brazos_ab, piernas_ab)
                    elif nueva:                 # nadie en cuadro: se limpia el estado y se avisa a TouchDesigner
                        prev.clear()
                        clasif and clasif.reiniciar()
                        if self.osc:
                            self._enviar_cero(cliente)
                        self.stats.update(visibilidad=0.0, brazos=None, piernas=None, clase=None)
                else:                           # deteccion apagada desde la interfaz: no se envia nada
                    ult_pts, n_cuadro = None, 0
                    prev.clear()
                    clasif and clasif.reiniciar()
                    if not enviado_cero and self.osc:
                        self._enviar_cero(cliente)
                    enviado_cero = True
                    self.stats.update(visibilidad=0.0, brazos=None, piernas=None, clase=None)

                # --- publicar el cuadro y actualizar cifras (FPS y ms/cuadro con media movil) ---
                self.stats["detectado"] = persona
                self.stats["cuadros"] += 1
                self.stats["con_persona"] += 1 if persona else 0
                self._publicar(vista)

                trabajo = time.perf_counter() - inicio
                if self.es_video:                          # reproducir a la velocidad real del video
                    resto = 1.0 / fps_video - trabajo
                    if resto > 0:
                        time.sleep(resto)
                if self.es_video and trabajo > 1.0 / fps_video:
                    # el PC va mas lento que el video: saltar cuadros para que se vea a velocidad real
                    for _ in range(min(10, int(trabajo * fps_video))):
                        cap.grab()
                total = time.perf_counter() - inicio
                fps_ema = (1 / total) if fps_ema == 0 else 0.9 * fps_ema + 0.1 * (1 / total)
                proc = time.perf_counter() - t_proc - (resto if (self.es_video and resto > 0) else 0.0)
                ms_ema = proc * 1000 if ms_ema == 0 else 0.9 * ms_ema + 0.1 * proc * 1000
                self.stats.update(fps=fps_ema, ms_cuadro=ms_ema, ms_clasificador=ms_clasif_ema)
        except Exception as e:
            self.error = f"Error en el motor: {e}"
        finally:                                # siempre: apagar OSC, cerrar el modelo y liberar la camara
            self.activo = False
            self.terminado = True
            try:
                self._enviar_cero(cliente)
            except Exception:
                pass
            marcador.close()
            cap.release()
            with self._cond:
                self._cond.notify_all()
