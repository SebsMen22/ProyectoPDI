"""
servidor.py  -  Interfaz de Shadow Dance
========================================
Servidor LOCAL (solo este computador, 127.0.0.1) que sirve la pagina de la interfaz y deja que ella controle
el motor. Usa solo la biblioteca estandar de Python.

  GET  /                  la interfaz (carpeta web\\)
  GET  /video             el video en vivo (MJPEG) con el esqueleto dibujado
  GET  /api/estado        cifras en vivo: FPS, deteccion, postura, barras, TouchDesigner
  GET  /api/videos        videos de prueba que encuentra
  GET  /api/resultados    metricas medidas (results\\sistema_metricas.csv y clasificador_metricas.csv)
  POST /api/iniciar       arranca la camara           {osc, detectar, clahe, camara}
  POST /api/reproducir    reproduce un video          {id, osc, detectar, clahe, bucle}
  POST /api/ajustes       cambia en caliente          {detectar, clahe, osc, bucle, pausado, reiniciar}
  POST /api/detener       para la camara o el video
  POST /api/salir         cierra la interfaz
"""
import csv
import glob
import json
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

from motor import Motor, RAIZ

AQUI = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(AQUI, "web")
ASSETS = os.path.join(AQUI, "assets")
EXT_VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class Estado:
    """Lo unico compartido entre las peticiones: el motor activo, el ultimo mensaje y el 'latido' de la pagina."""
    def __init__(self, cfg):
        """cfg = config.json; motor = el motor activo; ultimo_latido = ultima vez que la pagina pidio el estado."""
        self.cfg = cfg
        self.motor = None
        self.lock = threading.Lock()
        self.mensaje = ""
        self.ultimo_latido = None
        self.salir = threading.Event()
        self.videos = {}

    def detener_motor(self):
        """Detiene el motor activo (si lo hay) y lo suelta."""
        with self.lock:
            m, self.motor = self.motor, None
        if m:
            m.detener()


def listar_videos(estado):
    """Busca videos en las carpetas de config.json. Devuelve la lista y deja un id -> ruta en el estado."""
    salida, ids = [], {}
    for carpeta in estado.cfg.get("carpetas_videos", []):
        base = carpeta if os.path.isabs(carpeta) else os.path.join(RAIZ, carpeta)
        if not os.path.isdir(base):
            continue
        for ruta in sorted(glob.glob(os.path.join(base, "*"))):
            if not ruta.lower().endswith(EXT_VIDEO):
                continue
            vid = os.path.relpath(ruta, RAIZ).replace("\\", "/")
            info = {"id": vid, "nombre": os.path.basename(ruta), "carpeta": os.path.relpath(base, RAIZ).replace("\\", "/"),
                    "mb": round(os.path.getsize(ruta) / 1e6, 1), "duracion": None, "resolucion": None}
            cap = cv2.VideoCapture(ruta)
            if cap.isOpened():
                fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                info["duracion"] = round(n / fps, 1) if fps else None
                info["resolucion"] = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            cap.release()
            ids[vid] = ruta
            salida.append(info)
    estado.videos = ids
    return salida


def leer_csv(ruta):
    """Lee un CSV de results\\ como lista de diccionarios; lista vacia si todavia no existe."""
    if not os.path.isfile(ruta):
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hacer_manejador(estado):
    """Crea la clase que atiende las peticiones HTTP (queda ligada al 'estado' compartido)."""
    class Manejador(BaseHTTPRequestHandler):
        """GET: pagina, video y datos. POST: ordenes de la interfaz (iniciar, ajustes, detener)."""
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):          # sin ruido en la terminal
            pass

        # -------------------------------------------------------------------------- respuestas ---
        def _json(self, datos, codigo=200):
            """Responde con JSON."""
            cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def _archivo(self, ruta):
            """Responde con el contenido de un archivo (HTML, CSS, JS, PNG...)."""
            try:
                with open(ruta, "rb") as f:
                    datos = f.read()
            except OSError:
                return self._json({"error": "no existe"}, 404)
            tipo = mimetypes.guess_type(ruta)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", tipo + ("; charset=utf-8" if tipo.startswith("text/") or tipo.endswith("javascript") else ""))
            self.send_header("Content-Length", str(len(datos)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(datos)

        def _cuerpo(self):
            """Lee el JSON que manda la pagina en un POST (maximo 64 KB; si algo falla, un dict vacio)."""
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 65536:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                return {}

        def _estatico(self, base, rel):
            """Sirve un archivo de 'base' sin dejar salir de esa carpeta."""
            ruta = os.path.normpath(os.path.join(base, rel))
            if os.path.commonpath([ruta, base]) != base or not os.path.isfile(ruta):
                return self._json({"error": "no existe"}, 404)
            self._archivo(ruta)

        # ------------------------------------------------------------------------------- GET ---
        def do_GET(self):
            """Enruta las peticiones GET: pagina, video en vivo, estado, lista de videos y resultados."""
            ruta = urlparse(self.path).path
            if ruta in ("/", "/index.html"):
                return self._archivo(os.path.join(WEB, "index.html"))
            if ruta == "/video":
                return self._video()
            if ruta == "/api/estado":
                estado.ultimo_latido = time.time()
                m = estado.motor
                datos = m.estado() if m else {"activo": False, "modo": None, "error": None}
                datos.update(mensaje=estado.mensaje,
                             camara=estado.cfg.get("camara"), osc_destino=f"{estado.cfg.get('osc_ip', '127.0.0.1')}:7000")
                return self._json(datos)
            if ruta == "/api/videos":
                return self._json(listar_videos(estado))
            if ruta == "/api/resultados":
                res = os.path.join(RAIZ, "results")
                return self._json({"sistema": leer_csv(os.path.join(res, "sistema_metricas.csv")),
                                   "clasificador": leer_csv(os.path.join(res, "clasificador_metricas.csv"))})
            if ruta.startswith("/assets/"):
                return self._estatico(ASSETS, ruta[len("/assets/"):])
            return self._estatico(WEB, ruta.lstrip("/"))

        def _video(self):
            """Transmision MJPEG: manda cada cuadro nuevo del motor mientras la pagina siga conectada."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            motor, ultimo = estado.motor, -1
            try:
                while motor is not None and motor is estado.motor and not estado.salir.is_set():
                    jpeg, ultimo2 = motor.esperar_cuadro(ultimo, 1.0)
                    if jpeg is None:
                        if motor.terminado and not motor.activo:
                            break
                        continue
                    ultimo = ultimo2
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            self.close_connection = True

        # ------------------------------------------------------------------------------ POST ---
        def do_POST(self):
            """Enruta las ordenes de la interfaz: iniciar, reproducir, ajustes en caliente, detener y salir."""
            ruta, datos = urlparse(self.path).path, self._cuerpo()
            if ruta == "/api/iniciar":
                return self._iniciar(datos)
            if ruta == "/api/reproducir":
                return self._reproducir(datos)
            if ruta == "/api/ajustes":
                m = estado.motor
                if m is None:
                    return self._json({"ok": False})
                for clave in ("detectar", "clahe", "osc", "bucle", "pausado"):
                    if clave in datos:
                        setattr(m, clave, bool(datos[clave]))
                if datos.get("reiniciar"):
                    m.reiniciar_video = True
                    m.pausado = False
                return self._json({"ok": True})
            if ruta == "/api/detener":
                estado.detener_motor()
                estado.mensaje = ""
                return self._json({"ok": True})
            if ruta == "/api/salir":
                estado.detener_motor()
                estado.salir.set()
                return self._json({"ok": True})
            return self._json({"error": "ruta desconocida"}, 404)

        def _arrancar(self, motor):
            """Para el motor anterior (si hay) y arranca el nuevo."""
            estado.detener_motor()
            with estado.lock:
                estado.motor = motor
            motor.iniciar()

        def _iniciar(self, d):
            """Pantalla Ejecutar: arranca el motor con la camara elegida."""
            cfg = dict(estado.cfg)
            if "camara" in d:
                try:
                    cfg["camara"] = max(0, min(9, int(d["camara"])))
                    estado.cfg["camara"] = cfg["camara"]
                except (TypeError, ValueError):
                    pass
            estado.mensaje = ""
            self._arrancar(Motor(cfg, None, osc=d.get("osc", True), detectar=d.get("detectar", True),
                                 clahe=d.get("clahe", False)))
            return self._json({"ok": True, "mensaje": estado.mensaje})

        def _reproducir(self, d):
            """Pantalla Pruebas: arranca el motor con un video de la lista."""
            if not estado.videos:
                listar_videos(estado)
            ruta = estado.videos.get(d.get("id"))          # solo videos de la lista: no se abre cualquier ruta
            if not ruta:
                return self._json({"ok": False, "error": "video no encontrado"}, 404)
            estado.mensaje = ""
            self._arrancar(Motor(estado.cfg, ruta, osc=bool(d.get("osc", False)), detectar=d.get("detectar", True),
                                 clahe=d.get("clahe", False), bucle=d.get("bucle", True)))
            return self._json({"ok": True})

    return Manejador


def crear_servidor(cfg):
    """Crea el servidor en 127.0.0.1 (solo este computador). Devuelve (servidor, estado)."""
    estado = Estado(cfg)
    servidor = ThreadingHTTPServer(("127.0.0.1", int(cfg.get("puerto", 8765))), hacer_manejador(estado))
    servidor.daemon_threads = True
    return servidor, estado
