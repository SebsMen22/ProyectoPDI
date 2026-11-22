"""
evaluar_sistema.py  -  Shadow Dance
===================================
Mide el sistema con VIDEOS reales (asi la prueba se puede repetir igual cuantas veces quieras) y bajo
cuatro condiciones de imagen, para ver cuanto ayuda el procesamiento digital de imagenes (PDI):

    normal            la imagen tal cual
    clahe             ecualizacion adaptativa del histograma (CLAHE) sobre la luminosidad, ANTES de MediaPipe
    oscuro xF         imagen oscurecida a F veces su brillo (simula poca luz); por defecto F = 0.1, 0.06 y 0.04
    oscuro xF + clahe la misma imagen oscura, pero con CLAHE
Asi se ve a que nivel de oscuridad deja de detectar el sistema y cuanto lo rescata el procesamiento de imagen.

Por cada video y condicion mide:
    - % de cuadros con persona detectada
    - tiempo de procesar cada cuadro (preproceso + MediaPipe + clasificador): media y percentil 95
    - FPS que eso permite (1000 / tiempo medio)
    - tiempo del clasificador de postura
    - si le dices que clase deberia salir (video:clase), % de cuadros con la postura correcta
    - cuantos cuadros salio cada postura

Uso (desde la raiz del proyecto):
    python src\\evaluar_sistema.py --videos clips\\quieto.mp4:ninguna clips\\baile.mp4 lab_deteccion\\video_brazos.mp4
Despues del nombre del video puedes poner :brazos, :piernas, :ambas o :ninguna (la postura que se mantiene).
Salidas: results\\sistema_metricas.csv, results\\sistema_resumen.md y results\\sistema_oscuridad.png
(no guarda imagenes de personas salvo con --guardar-imagenes).
Los FPS dependen de TU computador: corre esto en el equipo de la demo.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postura import ClasificadorVivo  # noqa: E402

CLASES = ("brazos", "piernas", "ambas", "ninguna")
MODELO = "pose_landmarker_full.task"
_AQUI = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- funciones puras ---
def oscurecer(img, factor=0.3):
    """Multiplica el brillo de la imagen BGR por 'factor' (0.3 = 30 % del brillo original)."""
    import cv2
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def ecualizar_clahe(img, limite=2.0, bloque=8):
    """CLAHE: ecualizacion de histograma por zonas, aplicada solo a la luminosidad (canal L de LAB)."""
    import cv2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=limite, tileGridSize=(bloque, bloque)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def lista_condiciones(factores):
    """[(nombre, factor_de_brillo o None, usa_clahe), ...]: normal, clahe y, por cada factor, oscuro y oscuro + clahe."""
    conds = [("normal", None, False), ("clahe", None, True)]
    for f in factores:
        conds += [(f"oscuro x{f:g}", f, False), (f"oscuro x{f:g} + clahe", f, True)]
    return conds


def aplicar_condicion(img, factor, usar_clahe):
    """Aplica a un cuadro la condicion de prueba: primero oscurece (si hay factor) y luego CLAHE (si se pide)."""
    if factor is not None:
        img = oscurecer(img, factor)
    if usar_clahe:
        img = ecualizar_clahe(img)
    return img


def separar_clase(texto):
    """'clips\\baile.mp4:brazos' -> ('clips\\baile.mp4', 'brazos'); sin ':clase' -> (texto, None)."""
    if ":" in texto:
        ruta, resto = texto.rsplit(":", 1)
        if resto in CLASES:
            return ruta, resto
    return texto, None


def estadisticas(ms):
    """media y percentil 95 de una lista de tiempos (ms)."""
    if not len(ms):
        return 0.0, 0.0
    return float(np.mean(ms)), float(np.percentile(ms, 95))


# ------------------------------------------------------------------------------------ medicion ---
def buscar_modelo(ruta):
    """Ruta de pose_landmarker_full.task: la dada, la de src\\ o la carpeta actual; si no esta, termina con un aviso."""
    for c in (ruta, os.path.join(_AQUI, MODELO), MODELO):
        if c and os.path.isfile(c):
            return c
    sys.exit(f"No encuentro {MODELO}. Ponlo en src\\ o usa --modelo ruta")


def evaluar_video(ruta, condicion, ruta_modelo, ruta_postura, ancho_proceso, max_cuadros, esperada,
                  guardar_dir=None):
    """Procesa un video completo bajo una condicion y devuelve una fila de metricas (deteccion, tiempos, postura)."""
    nombre_cond, factor, usar_clahe = condicion
    import cv2
    import mediapipe as mp
    # --- abrir el video y crear el detector y el clasificador (mismos ajustes que pose_osc.py) ---
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        sys.exit(f"No pude abrir el video {ruta}")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opciones = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ruta_modelo),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
    clasif = ClasificadorVivo(ruta_postura, 5, 3) if ruta_postura else None   # mismos valores que pose_osc.py

    # --- recorrer los cuadros midiendo tiempos ---
    t_total, t_mp, n, detectados = [], [], 0, 0
    posturas = Counter()
    with mp.tasks.vision.PoseLandmarker.create_from_options(opciones) as det:
        while n < max_cuadros:
            ok, frame = cap.read()
            if not ok:
                break
            if ancho_proceso and frame.shape[1] != ancho_proceso:      # igual que la camara en vivo (640 de ancho)
                alto = int(round(frame.shape[0] * ancho_proceso / frame.shape[1]))
                frame = cv2.resize(frame, (ancho_proceso, alto), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()                                   # t0..t3: marcas para medir cada etapa
            img = aplicar_condicion(frame, factor, usar_clahe)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t1 = time.perf_counter()
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(n * 1000 / fps_video))   # MediaPipe
            t2 = time.perf_counter()
            if res.pose_landmarks:                                     # hay persona: se cuenta y se clasifica la postura
                detectados += 1
                if clasif:
                    posturas[clasif.actualizar([(p.x, p.y) for p in res.pose_landmarks[0]])[2]] += 1
            elif clasif:
                clasif.reiniciar()
            t3 = time.perf_counter()
            t_total.append((t3 - t0) * 1000)
            t_mp.append((t2 - t1) * 1000)
            if guardar_dir and n % 60 == 0:
                os.makedirs(guardar_dir, exist_ok=True)
                cv2.imwrite(os.path.join(guardar_dir, f"{os.path.basename(ruta)}_{nombre_cond.replace(' + ', '_')}_{n:04d}.jpg"), img)
            n += 1
    cap.release()

    # --- resumen del video: porcentajes y estadisticas de tiempo ---
    media, p95 = estadisticas(t_total)
    mp_media, _ = estadisticas(t_mp)
    fila = {
        "video": os.path.basename(ruta), "condicion": nombre_cond, "factor_brillo": factor if factor is not None else 1.0,
        "clahe": int(usar_clahe), "cuadros": n,
        "deteccion_pct": 100 * detectados / n if n else 0.0,
        "ms_cuadro_media": media, "ms_cuadro_p95": p95, "ms_mediapipe": mp_media,
        "ms_clasificador": clasif.ms_promedio() if clasif else float("nan"),
        "fps_posibles": 1000 / media if media else 0.0,
        "clase_esperada": esperada or "",
        "postura_correcta_pct": (100 * posturas[esperada] / sum(posturas.values())
                                 if esperada and sum(posturas.values()) else float("nan")),
        "posturas": " ".join(f"{k}:{v}" for k, v in posturas.most_common()),
    }
    return fila


def dibujar_oscuridad(filas, ruta):
    """Deteccion (%) segun el brillo de la imagen, sin y con CLAHE. Una linea por video y por condicion."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for video in sorted({r["video"] for r in filas}):
        for clahe, estilo, etiqueta in ((0, "o-", "sin CLAHE"), (1, "s--", "con CLAHE")):
            pts = sorted((r["factor_brillo"], r["deteccion_pct"]) for r in filas if r["video"] == video and r["clahe"] == clahe)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], estilo, label=f"{video} ({etiqueta})")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Brillo de la imagen (1 = original; mas a la derecha = mas oscuro)")
    ax.set_ylabel("Cuadros con persona detectada (%)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    """Lee los argumentos, evalua cada video en cada condicion y guarda el CSV, el resumen .md y la grafica."""
    ap = argparse.ArgumentParser(description="Evalua el sistema con videos y condiciones de imagen")
    ap.add_argument("--videos", nargs="+", required=True, help="videos; opcional ':clase' al final de cada uno")
    ap.add_argument("--modelo", default=None, help="pose_landmarker_full.task (por defecto busca en src\\)")
    ap.add_argument("--postura", default=os.path.join("modelos", "postura.joblib"), help="modelo de postura entrenado")
    ap.add_argument("--salida", default="results")
    ap.add_argument("--ancho", type=int, default=640, help="ancho al que se reduce cada cuadro (como la camara en vivo)")
    ap.add_argument("--max-cuadros", type=int, default=900, help="maximo de cuadros por video")
    ap.add_argument("--oscuro", type=float, nargs="*", default=[0.1, 0.06, 0.04],
                    help="factores de brillo a probar (0.1 = 10 %% del brillo original)")
    ap.add_argument("--guardar-imagenes", action="store_true",
                    help="guarda 1 cuadro de cada 60 en results\\evidencia (cuidado: pueden verse personas)")
    args = ap.parse_args()

    # --- preparar: modelo de postura (opcional) y modelo de pose ---
    ruta_post = args.postura if os.path.isfile(args.postura) else None
    if ruta_post is None:
        print("(no encuentro el modelo de postura: se mide sin clasificador)")
    modelo = buscar_modelo(args.modelo)
    # --- medir cada video en cada condicion ---
    filas = []
    for texto in args.videos:
        ruta, esperada = separar_clase(texto)
        for cond in lista_condiciones(args.oscuro):
            print(f"-> {os.path.basename(ruta)} | {cond[0]} ...", flush=True)
            dir_img = os.path.join(args.salida, "evidencia") if args.guardar_imagenes else None
            filas.append(evaluar_video(ruta, cond, modelo, ruta_post, args.ancho, args.max_cuadros, esperada,
                                       dir_img))

    # --- guardar resultados: CSV, tabla Markdown y grafica ---
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "sistema_metricas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    lin = ["| Video | Condicion | Cuadros | Deteccion | ms/cuadro (media / p95) | ms MediaPipe | ms clasif. | FPS posibles | Postura correcta |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in filas:
        ok = "-" if r["postura_correcta_pct"] != r["postura_correcta_pct"] else f"{r['postura_correcta_pct']:.0f} % ({r['clase_esperada']})"
        cl = "-" if r["ms_clasificador"] != r["ms_clasificador"] else f"{r['ms_clasificador']:.1f}"
        lin.append(f"| {r['video']} | {r['condicion']} | {r['cuadros']} | {r['deteccion_pct']:.1f} % | "
                   f"{r['ms_cuadro_media']:.1f} / {r['ms_cuadro_p95']:.1f} | {r['ms_mediapipe']:.1f} | {cl} | "
                   f"{r['fps_posibles']:.0f} | {ok} |")
    texto_md = "\n".join(lin)
    with open(os.path.join(args.salida, "sistema_resumen.md"), "w", encoding="utf-8") as f:
        f.write(texto_md + "\n")
    print("\n" + texto_md)
    dibujar_oscuridad(filas, os.path.join(args.salida, "sistema_oscuridad.png"))
    print(f"\nGuardado en {args.salida}\\sistema_metricas.csv y sistema_resumen.md")


if __name__ == "__main__":
    main()
