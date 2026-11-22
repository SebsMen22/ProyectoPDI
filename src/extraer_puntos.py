"""
extraer_puntos.py  -  Shadow Dance
==================================
Lee una carpeta de fotos y guarda en un CSV los 33 puntos del cuerpo (MediaPipe) de cada una,
junto con su etiqueta y un "grupo" para separar entrenamiento y prueba sin trampa.
SOLO escribe numeros (coordenadas): no copia ni modifica las fotos, y el CSV no contiene imagenes.

Etiquetas (se calculan solas, sin etiquetar a mano):
  - Fotos 1..LIMITE  -> clase 'brazos'    (la categoria que definio el laboratorio)
  - Fotos LIMITE+1.. -> clase 'piernas'   (piernas_abiertas en el laboratorio)
  - Si la MISMA imagen (mismo contenido exacto, comparado con un hash) aparece en las dos
    categorias, es una foto con brazos y piernas abiertos: clase 'ambas'.

Grupos (para no mezclar fotos casi iguales entre entrenamiento y prueba):
  - Las fotos se agrupan en bloques de BLOQUE numeros consecutivos (rafagas de una misma sesion).
  - Los bloques que comparten una imagen repetida se unen en un solo grupo.
  - La columna 'persona' dice quien aparece: 'A' por defecto, 'B' para los numeros de --persona-b.

Uso (Windows, desde la raiz del proyecto):
    python src\\extraer_puntos.py --fotos lab_deteccion\\dataset --salida data\\keypoints.csv
    python src\\extraer_puntos.py --fotos lab_deteccion\\dataset --salida data\\keypoints.csv --persona-b 179,180
    python src\\extraer_puntos.py --fotos lab_deteccion\\dataset --salida data\\etiquetas.csv --solo-etiquetas   (sin MediaPipe)
"""
import argparse
import csv
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

N_PUNTOS = 33
CAMPOS = ("x", "y", "z", "v")   # coordenada normalizada x, y, profundidad relativa z y visibilidad


# ----------------------------------------------------------------------------- funciones puras ---
def numero_de_foto(nombre):
    """'Img  (28).jpg' -> 28. Devuelve None si el nombre no trae un numero entre parentesis."""
    m = re.search(r"\((\d+)\)", nombre)
    return int(m.group(1)) if m else None


def parsear_numeros(texto):
    """'1-5,179,180' -> {1,2,3,4,5,179,180}. Texto vacio -> conjunto vacio."""
    salida = set()
    for trozo in (texto or "").replace(" ", "").split(","):
        if not trozo:
            continue
        if "-" in trozo:
            a, b = trozo.split("-", 1)
            salida.update(range(int(a), int(b) + 1))
        else:
            salida.add(int(trozo))
    return salida


def hash_de_archivo(ruta):
    """Huella MD5 del contenido de un archivo: dos fotos identicas dan el mismo valor."""
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


class _Union:
    """Union-find minimo para juntar bloques que comparten imagenes repetidas."""
    def __init__(self):
        self.padre = {}

    def buscar(self, x):
        """Representante del grupo de x (con compresion de camino)."""
        self.padre.setdefault(x, x)
        while self.padre[x] != x:
            self.padre[x] = self.padre[self.padre[x]]
            x = self.padre[x]
        return x

    def unir(self, a, b):
        """Junta los grupos de a y b en uno solo (queda el menor como representante)."""
        ra, rb = self.buscar(a), self.buscar(b)
        if ra != rb:
            self.padre[max(ra, rb)] = min(ra, rb)


def etiquetar(fotos, limite=90, bloque=10, persona_b=frozenset()):
    """
    fotos: lista de (numero, hash). Devuelve un diccionario numero -> {clase, clase_original, grupo, persona}.
    Funcion pura: no toca archivos, asi se puede probar sin fotos.
    """
    clase_orig = {n: ("brazos" if n <= limite else "piernas") for n, _ in fotos}
    clases_por_hash = defaultdict(set)
    for n, h in fotos:
        clases_por_hash[h].add(clase_orig[n])

    # Bloque de cada foto: se cuenta dentro de su categoria original (asi 1-10, 11-20... y 91-100, ...)
    # y nunca mezcla personas distintas (el nombre del bloque lleva la letra de la persona).
    def bloque_de(n):
        """Nombre del bloque de la foto n: persona + categoria + numero de bloque (ej. 'Ab0')."""
        base = 0 if n <= limite else limite
        persona = "B" if n in persona_b else "A"
        return f"{persona}{clase_orig[n][0]}{(n - 1 - base) // bloque}"

    uf = _Union()
    por_hash = defaultdict(list)
    for n, h in fotos:
        uf.buscar(bloque_de(n))
        por_hash[h].append(n)
    for _, nums in por_hash.items():
        for otro in nums[1:]:
            uf.unir(bloque_de(nums[0]), bloque_de(otro))   # imagen repetida: sus bloques van juntos

    salida = {}
    for n, h in fotos:
        repetida_en_ambas = len(clases_por_hash[h]) > 1
        salida[n] = {
            "clase": "ambas" if repetida_en_ambas else clase_orig[n],
            "clase_original": clase_orig[n],
            "grupo": uf.buscar(bloque_de(n)),
            "persona": "B" if n in persona_b else "A",
            "hash": h[:10],
        }
    return salida


# --------------------------------------------------------------------------------- MediaPipe ---
def buscar_modelo(ruta):
    """Ruta de pose_landmarker_full.task (la dada, src\\ o la carpeta actual); si no esta, termina con un aviso."""
    for candidato in (ruta, os.path.join("src", "pose_landmarker_full.task"), "pose_landmarker_full.task"):
        if candidato and os.path.isfile(candidato):
            return candidato
    sys.exit("No encuentro pose_landmarker_full.task. Ponlo en src\\ o indica --modelo ruta")


def extraer_puntos(filas, ruta_modelo):
    """Agrega a cada fila 'detectada' (0/1) y 'puntos' (lista de 33 tuplas x,y,z,v)."""
    import cv2
    import mediapipe as mp
    opciones = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ruta_modelo),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,   # fotos sueltas: no hay seguimiento entre cuadros
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(opciones) as detector:
        for i, fila in enumerate(filas, 1):
            img = cv2.imread(fila["ruta"])
            fila["detectada"], fila["puntos"] = 0, None
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if res.pose_landmarks:
                    fila["detectada"] = 1
                    fila["puntos"] = [(p.x, p.y, p.z, p.visibility) for p in res.pose_landmarks[0]]
            if i % 20 == 0 or i == len(filas):
                print(f"  {i}/{len(filas)} fotos procesadas")


# ------------------------------------------------------------------------------------- main ---
def main():
    """Lee las fotos, calcula etiquetas y grupos, extrae los puntos con MediaPipe y escribe el CSV."""
    ap = argparse.ArgumentParser(description="Extrae los 33 puntos del cuerpo de cada foto y los guarda en un CSV")
    ap.add_argument("--fotos", required=True, help="carpeta con las fotos (por ejemplo lab_deteccion\\dataset)")
    ap.add_argument("--salida", default=os.path.join("data", "keypoints.csv"), help="CSV de salida")
    ap.add_argument("--modelo", default=None, help="ruta de pose_landmarker_full.task (por defecto busca en src\\)")
    ap.add_argument("--limite", type=int, default=90, help="ultima foto de la clase 'brazos' (por defecto 90)")
    ap.add_argument("--bloque", type=int, default=10, help="fotos consecutivas que se consideran una misma sesion")
    ap.add_argument("--persona-b", default="", help="numeros de las fotos de la segunda persona, ej. 1-5,179,180")
    ap.add_argument("--solo-etiquetas", action="store_true", help="solo calcula etiquetas y grupos (no usa MediaPipe)")
    args = ap.parse_args()

    # --- 1. listar las fotos y calcular el hash de cada una ---
    archivos = sorted(f for f in os.listdir(args.fotos) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    fotos, rutas = [], {}
    for f in archivos:
        n = numero_de_foto(f)
        if n is None:
            print(f"  (se omite {f}: el nombre no trae un numero entre parentesis)")
            continue
        ruta = os.path.join(args.fotos, f)
        fotos.append((n, hash_de_archivo(ruta)))
        rutas[n] = (f, ruta)
    if not fotos:
        sys.exit(f"No encontre fotos con nombre tipo 'Img (1).jpg' en {args.fotos}")
    fotos.sort()

    # --- 2. etiquetas y grupos (sin MediaPipe) ---
    etiquetas = etiquetar(fotos, args.limite, args.bloque, parsear_numeros(args.persona_b))
    filas = []
    for n, _ in fotos:
        fila = {"archivo": rutas[n][0], "numero": n, "ruta": rutas[n][1], **etiquetas[n]}
        filas.append(fila)

    print(f"{len(filas)} fotos | clases: {dict(Counter(f['clase'] for f in filas))} | "
          f"grupos: {len({f['grupo'] for f in filas})} | personas: {dict(Counter(f['persona'] for f in filas))}")

    # --- 3. puntos del cuerpo con MediaPipe ---
    if not args.solo_etiquetas:
        extraer_puntos(filas, buscar_modelo(args.modelo))

    # --- 4. escribir el CSV: datos de la foto + 33 puntos x (x, y, z, visibilidad) ---
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    columnas = ["archivo", "numero", "clase", "clase_original", "grupo", "persona", "hash", "detectada"]
    columnas += [f"p{i}_{c}" for i in range(N_PUNTOS) for c in CAMPOS]
    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columnas)
        for fila in filas:
            base = [fila["archivo"], fila["numero"], fila["clase"], fila["clase_original"], fila["grupo"],
                    fila["persona"], fila["hash"], fila.get("detectada", "")]
            pts = fila.get("puntos")
            w.writerow(base + ([v for p in pts for v in p] if pts else [""] * (N_PUNTOS * len(CAMPOS))))
    print(f"Guardado: {args.salida}")

    # --- 5. resumen de deteccion por clase ---
    if not args.solo_etiquetas:
        print("\n===== RESUMEN DE DETECCION =====")
        for clase in sorted({f['clase'] for f in filas}):
            sub = [f for f in filas if f["clase"] == clase]
            ok = sum(f["detectada"] for f in sub)
            print(f"  {clase:9s}: {ok}/{len(sub)} fotos con persona detectada ({100 * ok / len(sub):.1f} %)")
        print("Las fotos sin deteccion quedan en el CSV con detectada = 0 y se excluyen al entrenar.")


if __name__ == "__main__":
    main()
