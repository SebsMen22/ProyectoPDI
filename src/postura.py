"""
postura.py  -  Shadow Dance
===========================
Geometria del cuerpo para clasificar la postura a partir de los 33 puntos de MediaPipe.
Este modulo lo usan dos programas: el que ENTRENA (entrenar_clasificador.py) y, mas adelante,
el que clasifica EN VIVO (pose_osc.py). Asi las dos partes calculan las mismas caracteristicas.

Dos preguntas de si o no sobre cada persona:
    - abre_brazos  : los brazos estan abiertos / levantados
    - abre_piernas : las piernas estan abiertas
Combinadas dan: 'brazos', 'piernas', 'ambas' o 'ninguna'.

Todas las medidas se dividen por el largo del TRONCO (del medio de los hombros al medio de las caderas),
asi no dependen de que la persona este cerca o lejos de la camara, ni de si es adulta o pequena.
"""
import math

import numpy as np

# Indices de MediaPipe Pose
HOMBRO_I, HOMBRO_D = 11, 12
CODO_I, CODO_D = 13, 14
MUNECA_I, MUNECA_D = 15, 16
CADERA_I, CADERA_D = 23, 24
RODILLA_I, RODILLA_D = 25, 26
TOBILLO_I, TOBILLO_D = 27, 28

NOMBRES = [
    "munecas_distancia",        # separacion entre las dos munecas
    "munecas_altura_media",     # cuanto estan las munecas por encima de los hombros (promedio)
    "munecas_altura_max",       # la muneca mas alta
    "brazo_extension",          # distancia hombro-muneca (promedio)
    "codo_angulo",              # angulo del codo, 0 a 1 (1 = brazo estirado)
    "hombros_ancho",            # ancho de hombros (cambia si la persona esta de lado)
    "tobillos_distancia",       # separacion entre los tobillos
    "rodillas_distancia",       # separacion entre las rodillas
    "rodilla_angulo",           # angulo de la rodilla, 0 a 1 (1 = pierna estirada)
    "tobillos_vs_hombros",      # separacion de tobillos comparada con el ancho de hombros
    "tobillos_altura",          # que tan abajo estan los tobillos respecto a la cadera
]


def _xy(puntos, i):
    """Coordenadas (x, y) del punto i de MediaPipe como decimales."""
    p = puntos[i]
    return float(p[0]), float(p[1])


def _dist(a, b):
    """Distancia entre dos puntos (x, y)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angulo(a, b, c):
    """Angulo en b (entre los segmentos b-a y b-c), llevado a 0..1 dividiendo por 180 grados."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos)) / 180.0


def caracteristicas(puntos):
    """
    puntos: 33 tuplas/listas (x, y, ...) con coordenadas normalizadas de MediaPipe (y crece hacia abajo).
    Devuelve un arreglo numpy con len(NOMBRES) valores.
    """
    # --- los puntos que se necesitan (izquierda / derecha) ---
    hi, hd = _xy(puntos, HOMBRO_I), _xy(puntos, HOMBRO_D)
    ci, cd = _xy(puntos, CODO_I), _xy(puntos, CODO_D)
    mi, md = _xy(puntos, MUNECA_I), _xy(puntos, MUNECA_D)
    cai, cad = _xy(puntos, CADERA_I), _xy(puntos, CADERA_D)
    ri, rd = _xy(puntos, RODILLA_I), _xy(puntos, RODILLA_D)
    ti, td = _xy(puntos, TOBILLO_I), _xy(puntos, TOBILLO_D)

    # --- escala: todo se divide por el largo del tronco (y por el ancho de hombros en una medida) ---
    h_med = ((hi[0] + hd[0]) / 2, (hi[1] + hd[1]) / 2)
    c_med = ((cai[0] + cad[0]) / 2, (cai[1] + cad[1]) / 2)
    tronco = max(_dist(h_med, c_med), 1e-6)
    ancho_h = max(_dist(hi, hd), 1e-6)

    # --- las 11 medidas, en el mismo orden que NOMBRES ---
    alt_i = (hi[1] - mi[1]) / tronco       # positivo = muneca por encima del hombro
    alt_d = (hd[1] - md[1]) / tronco
    return np.array([
        _dist(mi, md) / tronco,
        (alt_i + alt_d) / 2,
        max(alt_i, alt_d),
        (_dist(hi, mi) + _dist(hd, md)) / 2 / tronco,
        (_angulo(hi, ci, mi) + _angulo(hd, cd, md)) / 2,
        ancho_h / tronco,
        _dist(ti, td) / tronco,
        _dist(ri, rd) / tronco,
        (_angulo(cai, ri, ti) + _angulo(cad, rd, td)) / 2,
        min(_dist(ti, td) / ancho_h, 10.0),
        (((ti[1] + td[1]) / 2) - c_med[1]) / tronco,
    ], dtype=float)


CLASE_IDX = {"ninguna": 0, "brazos": 1, "piernas": 2, "ambas": 3}   # numero que viaja por OSC en /posture


class ClasificadorVivo:
    """
    Clasifica la postura cuadro a cuadro con el modelo guardado por entrenar_clasificador.py --guardar.
    - Solo clasifica 1 de cada 'cada' cuadros (ahorra tiempo; el resultado se mantiene entre medias).
    - Suaviza con una VOTACION de las ultimas 'ventana' predicciones de cada pregunta, para que la postura
      no parpadee por un cuadro dudoso.
    """
    def __init__(self, ruta, ventana=5, cada=2):
        """Carga el modelo (joblib) y prepara el historial de votos de cada pregunta."""
        import time
        from collections import deque
        import joblib
        datos = joblib.load(ruta)
        self.modelos, self.tipo = datos["modelos"], datos.get("tipo", "?")
        # Random forest: en vivo basta con 100 arboles (3x mas rapido, casi el mismo resultado que 300)
        for m in self.modelos.values():
            if hasattr(m, "estimators_") and len(m.estimators_) > 100:
                m.estimators_ = m.estimators_[:100]
                m.n_estimators = 100
        self.cada = max(1, cada)
        self.hist = {k: deque(maxlen=max(1, ventana)) for k in self.modelos}
        self._reloj = time.perf_counter
        self.reiniciar()
        self.ms_total, self.n_pred = 0.0, 0

    def reiniciar(self):
        """Llamar cuando no hay persona: olvida las votaciones anteriores."""
        for h in self.hist.values():
            h.clear()
        self.n = 0
        self.estado = (0, 0)

    @staticmethod
    def _votar(historial, anterior):
        """Voto de mayoria (1 o 0) de las ultimas predicciones; en empate conserva la respuesta anterior."""
        s, n = sum(historial), len(historial)
        return 1 if 2 * s > n else (0 if 2 * s < n else anterior)   # empate: se queda como estaba

    def actualizar(self, puntos):
        """puntos: 33 tuplas (x, y, ...) SIN espejo. Devuelve (abre_brazos, abre_piernas, nombre_de_clase)."""
        if self.n % self.cada == 0:             # solo se predice 1 de cada 'cada' cuadros
            t = self._reloj()
            x = caracteristicas(puntos).reshape(1, -1)
            for k, modelo in self.modelos.items():
                self.hist[k].append(int(modelo.predict(x)[0]))
            self.ms_total += (self._reloj() - t) * 1000
            self.n_pred += 1
            self.estado = (self._votar(self.hist["abre_brazos"], self.estado[0]),
                           self._votar(self.hist["abre_piernas"], self.estado[1]))
        self.n += 1
        return self.estado[0], self.estado[1], clase_desde_etiquetas(*self.estado)

    def ms_promedio(self):
        """Milisegundos promedio que tarda una prediccion (las dos preguntas juntas)."""
        return self.ms_total / self.n_pred if self.n_pred else 0.0


def etiquetas_desde_clase(clase):
    """'brazos' -> (1, 0); 'piernas' -> (0, 1); 'ambas' -> (1, 1); 'ninguna' -> (0, 0)."""
    return {"brazos": (1, 0), "piernas": (0, 1), "ambas": (1, 1), "ninguna": (0, 0)}[clase]


def clase_desde_etiquetas(abre_brazos, abre_piernas):
    """Inversa de etiquetas_desde_clase."""
    return {(1, 0): "brazos", (0, 1): "piernas", (1, 1): "ambas", (0, 0): "ninguna"}[(int(abre_brazos), int(abre_piernas))]
