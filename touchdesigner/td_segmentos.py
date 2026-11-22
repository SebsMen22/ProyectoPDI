"""
td_segmentos.py  -  Shadow Dance (se pega DENTRO de TouchDesigner, en script1_callbacks)
=======================================================================================
Dibuja una "cinta" de luz continua por cada miembro:
    brazo  : hombro - codo - muneca   (izquierdo y derecho)
    pierna : cadera - rodilla - tobillo (izquierda y derecha)
Cada tramo tiene degradado entre los colores de sus dos articulaciones.

Dos uniones:
  - EN EL ESPACIO: los puntos de un miembro se unen en una cinta (hombro-codo-muneca).
  - EN EL TIEMPO : entre la posicion del cuadro anterior y la actual se dibujan cintas INTERMEDIAS,
    para que al mover rapido el brazo quede un barrido continuo y no un abanico de lineas sueltas.

Regla de movimiento: la cadena COMPLETA se dibuja si CUALQUIER articulacion de la cadena
supera UMBRAL (asi, si solo mueves la muneca, igual se ve el brazo entero).

Lee los canales del nodo 'filter1': detected, lw_x, lw_y, lw_s, le_x ...
Necesita los 12 puntos (lw rw le re la ra lk rk ls rs lh rh): usa pose_osc.py / osc_simulador.py actualizados.
"""
import math
import numpy as np

# ------------------------------------------------------------------ Constantes que puedes ajustar ---
NODO_DATOS = 'filter1'     # nombre del Filter CHOP que ya tienes
ANCHO, ALTO = 1280, 720    # debe coincidir con la resolucion de tus circulos
UMBRAL = 0.15              # velocidad minima (anchos de imagen por segundo) para dibujar
GROSOR = 22                # radio de la cinta en pixeles (mas alto = cinta mas gruesa)
DIBUJAR_TRONCO = False     # True = tambien une hombro-hombro, cadera-cadera y hombro-cadera
PASO_PX = 14               # separacion maxima (px) entre cintas intermedias: menor = barrido mas lleno, pero mas lento
INTERP_MAX = 6             # maximo de cintas intermedias por cuadro (limita el costo si el FPS baja)

# Cadenas de articulaciones (se unen en orden). Nombres iguales a los de pose_osc.py
CADENAS = [('ls', 'le', 'lw'), ('rs', 're', 'rw'),
           ('lh', 'lk', 'la'), ('rh', 'rk', 'ra')]
TRONCO = [('ls', 'rs'), ('lh', 'rh'), ('ls', 'lh'), ('rs', 'rh')]

# Un color para los brazos y otro para las piernas (R, G, B entre 0 y 1).
# Evita colores con los tres canales altos: al sumarse en el feedback se queman a blanco.
COLOR_BRAZOS = (0.0, 0.85, 1.0)     # cian
COLOR_PIERNAS = (1.0, 0.1, 0.75)    # magenta
# La cinta es mas tenue en la raiz (hombro/cadera) y mas viva en la punta (muneca/tobillo): efecto de cola de luz.
BRILLO_RAIZ, BRILLO_MEDIO, BRILLO_PUNTA = 0.45, 0.75, 1.0


def _tono(color, k):
    return (color[0] * k, color[1] * k, color[2] * k)


COLORES = {}
for _lado in ('l', 'r'):
    COLORES[_lado + 's'] = _tono(COLOR_BRAZOS, BRILLO_RAIZ)     # hombro
    COLORES[_lado + 'e'] = _tono(COLOR_BRAZOS, BRILLO_MEDIO)    # codo
    COLORES[_lado + 'w'] = _tono(COLOR_BRAZOS, BRILLO_PUNTA)    # muneca
    COLORES[_lado + 'h'] = _tono(COLOR_PIERNAS, BRILLO_RAIZ)    # cadera
    COLORES[_lado + 'k'] = _tono(COLOR_PIERNAS, BRILLO_MEDIO)   # rodilla
    COLORES[_lado + 'a'] = _tono(COLOR_PIERNAS, BRILLO_PUNTA)   # tobillo

_previo = {}               # posicion (px) de cada articulacion en el cuadro anterior; persiste entre cuadros


def dibujar_segmentos(segs, ancho, alto, grosor):
    """
    Funcion pura (se puede probar sin TouchDesigner).
    segs: lista de (ax, ay, bx, by, color_a, color_b) con x,y en PIXELES; y = fila contada desde ABAJO.
    Devuelve un arreglo float32 (alto, ancho, 4) RGBA; la fila 0 es la de abajo (asi lo espera TouchDesigner).
    """
    img = np.zeros((alto, ancho, 4), dtype=np.float32)
    img[..., 3] = 1.0                                  # fondo negro opaco
    for ax, ay, bx, by, ca, cb in segs:
        # Solo se calcula dentro de la caja que rodea al segmento (mucho mas rapido que toda la imagen)
        x0 = int(max(0, min(ax, bx) - grosor - 1)); x1 = int(min(ancho, max(ax, bx) + grosor + 2))
        y0 = int(max(0, min(ay, by) - grosor - 1)); y1 = int(min(alto, max(ay, by) + grosor + 2))
        if x1 <= x0 or y1 <= y0:
            continue                                   # el segmento queda fuera de la imagen
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        bax, bay = bx - ax, by - ay
        largo2 = bax * bax + bay * bay
        pax, pay = xs - ax, ys - ay
        if largo2 < 1e-6:
            t = np.zeros_like(xs)                      # los dos puntos coinciden: es un punto
        else:
            t = np.clip((pax * bax + pay * bay) / largo2, 0.0, 1.0)   # posicion a lo largo del segmento
        dist = np.hypot(pax - t * bax, pay - t * bay)                 # distancia al segmento
        brillo = np.clip(1.0 - dist / grosor, 0.0, 1.0) ** 2          # borde suave
        color = (1 - t)[..., None] * np.array(ca, np.float32) + t[..., None] * np.array(cb, np.float32)
        capa = color * brillo[..., None]
        zona = img[y0:y1, x0:x1, :3]
        np.maximum(zona, capa, out=zona)               # donde se cruzan dos cintas gana la mas brillante
    return img


def tramos_a_dibujar(leer, cadenas, tronco, dibujar_tronco, umbral, ancho, alto,
                     previo=None, paso_px=PASO_PX, interp_max=INTERP_MAX):
    """
    Funcion pura (salvo que actualiza el diccionario 'previo', a proposito).
    'leer(nombre)' devuelve el valor de un canal, p. ej. leer('lw_x').
    Devuelve la lista de segmentos (en pixeles, y desde abajo) que hay que dibujar en este cuadro.
    Una cadena se dibuja completa si cualquiera de sus articulaciones supera el umbral.
    Si 'previo' trae la posicion del cuadro anterior, se agregan cintas intermedias entre las dos posiciones.
    """
    def punto(n):
        return leer(n + '_x') * ancho, (1 - leer(n + '_y')) * alto

    segs = []
    grupos = [c for c in cadenas]
    if dibujar_tronco:
        grupos += [tuple(par) for par in tronco]
    visto = {}                                          # posiciones actuales de todas las articulaciones leidas
    for miembros in grupos:
        try:
            actual = {n: punto(n) for n in miembros}
            visto.update(actual)
            if max(leer(n + '_s') for n in miembros) <= umbral:
                continue
        except Exception:
            continue                                    # falta algun canal: se salta esta cadena

        # Cuantas cintas intermedias hacen falta: segun lo que se movio la articulacion que mas se desplazo
        pasos = 1
        if previo is not None and all(n in previo for n in miembros):
            salto = max(math.hypot(actual[n][0] - previo[n][0], actual[n][1] - previo[n][1]) for n in miembros)
            if salto < 0.5 * ancho:                     # un salto enorme es un error de deteccion: no se rellena
                pasos = int(min(interp_max, max(1, math.ceil(salto / paso_px))))
        for k in range(1, pasos + 1):
            f = k / pasos
            pos = {}
            for n in miembros:
                if pasos > 1 and previo is not None and n in previo:
                    pos[n] = (previo[n][0] + f * (actual[n][0] - previo[n][0]),
                              previo[n][1] + f * (actual[n][1] - previo[n][1]))
                else:
                    pos[n] = actual[n]
            for a, b in zip(miembros[:-1], miembros[1:]):
                segs.append((pos[a][0], pos[a][1], pos[b][0], pos[b][1], COLORES[a], COLORES[b]))
    if previo is not None:
        previo.update(visto)                            # se guarda donde esta cada articulacion para el cuadro siguiente
    return segs


def onCook(scriptOp):
    _ = absTime.frame                                  # obliga a recalcular en cada cuadro
    datos = op(NODO_DATOS)
    if datos is None:
        return
    try:
        hay_persona = datos['detected'] > 0.5
    except Exception:
        hay_persona = False

    segs = []
    if hay_persona:
        segs = tramos_a_dibujar(lambda n: datos[n].eval(), CADENAS, TRONCO, DIBUJAR_TRONCO,
                                UMBRAL, ANCHO, ALTO, previo=_previo)
    else:
        _previo.clear()                                # al volver a entrar no se dibuja un barrido desde el pasado

    img = dibujar_segmentos(segs, ANCHO, ALTO, GROSOR)
    scriptOp.copyNumpyArray(np.ascontiguousarray(img))
