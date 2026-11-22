"""
generar_diagrama.py  -  Interfaz de Shadow Dance
================================================
Dibuja el diagrama "Como funciona" (assets\\como_funciona.png). La interfaz lo muestra, y sirve tambien para el
README y el poster. Solo necesita Pillow. Desde la raiz del proyecto:

    python INTERFAZ\\generar_diagrama.py
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "assets", "como_funciona.png")

FONDO, TARJETA, BORDE = (11, 11, 18), (22, 22, 36), (46, 46, 72)
TEXTO, SUAVE = (236, 236, 246), (139, 139, 167)
MORADO, CIAN, MAGENTA, ORO = (139, 92, 246), (34, 211, 238), (244, 114, 182), (251, 191, 36)
E = 2                                                     # se dibuja al doble y se reduce: bordes suaves


def fuente(tam, negrita=False):
    """Primera tipografia disponible (Windows, Linux o la que trae matplotlib)."""
    candidatos = (["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"] if negrita else
                  ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"])
    candidatos += ["/usr/share/fonts/opentype/inter/Inter-Bold.otf" if negrita else "/usr/share/fonts/opentype/inter/Inter-Regular.otf"]
    try:
        import matplotlib
        candidatos.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if negrita else "DejaVuSans.ttf"))
    except Exception:
        pass
    for c in candidatos:
        if os.path.isfile(c):
            return ImageFont.truetype(c, tam * E)
    return ImageFont.load_default()


def caja(d, x, y, w, h, numero, titulo, lineas, acento=MORADO):
    """Tarjeta del diagrama: fondo, franja de color, circulo con el numero del paso, titulo y lineas de texto."""
    d.rounded_rectangle([x * E, y * E, (x + w) * E, (y + h) * E], radius=18 * E, fill=TARJETA, outline=BORDE, width=2 * E)
    d.rounded_rectangle([x * E, y * E, (x + 6) * E, (y + h) * E], radius=3 * E, fill=acento)
    r = 15
    d.ellipse([(x + 24) * E, (y + 22) * E, (x + 24 + 2 * r) * E, (y + 22 + 2 * r) * E], fill=acento)
    f = fuente(17, True)
    d.text(((x + 24 + r) * E, (y + 22 + r) * E), str(numero), font=f, fill=(255, 255, 255), anchor="mm")
    d.text(((x + 66) * E, (y + 36) * E), titulo, font=fuente(21, True), fill=TEXTO, anchor="lm")
    for i, linea in enumerate(lineas):
        d.text(((x + 26) * E, (y + 84 + i * 28) * E), linea, font=fuente(16), fill=SUAVE, anchor="lm")


def flecha(d, x1, y1, x2, y2, color=MORADO):
    """Linea con punta de flecha (horizontal o vertical segun hacia donde apunte)."""
    d.line([x1 * E, y1 * E, x2 * E, y2 * E], fill=color, width=3 * E)
    if abs(x2 - x1) >= abs(y2 - y1):
        s = 1 if x2 > x1 else -1
        d.polygon([(x2 * E, y2 * E), ((x2 - 14 * s) * E, (y2 - 9) * E), ((x2 - 14 * s) * E, (y2 + 9) * E)], fill=color)
    else:
        s = 1 if y2 > y1 else -1
        d.polygon([(x2 * E, y2 * E), ((x2 - 9) * E, (y2 - 14 * s) * E), ((x2 + 9) * E, (y2 - 14 * s) * E)], fill=color)


def main():
    """Dibuja las 8 etapas, el puente OSC y la fila de TouchDesigner; guarda assets\\como_funciona.png."""
    W, H = 1800, 850
    img = Image.new("RGB", (W * E, H * E), FONDO)
    d = ImageDraw.Draw(img)

    d.text((60 * E, 52 * E), "Cómo funciona Shadow Dance", font=fuente(34, True), fill=TEXTO, anchor="lm")
    d.text((60 * E, 92 * E), "Una cámara ve tu cuerpo; el computador entiende la postura; la luz dibuja tu movimiento.",
           font=fuente(18), fill=SUAVE, anchor="lm")

    # Fila 1: Python
    d.text((60 * E, 150 * E), "PYTHON  ·  ve y entiende", font=fuente(15, True), fill=MORADO, anchor="lm")
    w, h, gap, x0, y1 = 380, 190, 56, 60, 175
    datos1 = [
        ("Cámara", ["USB (PS3 Eye)", "640 x 480 a 30 FPS", "o un video grabado"], MORADO),
        ("OpenCV  (PDI)", ["Lee cada cuadro", "Espejo y reducción de tamaño", "CLAHE opcional (poca luz)"], MORADO),
        ("MediaPipe Pose", ["Modelo preentrenado", "33 puntos del cuerpo", "hombros, codos, muñecas..."], MORADO),
        ("Clasificador de postura", ["11 medidas del cuerpo", "Random forest propio (fotos + capturas)", "Votación de los últimos cuadros"], CIAN),
    ]
    for i, (t, ls, ac) in enumerate(datos1):
        x = x0 + i * (w + gap)
        caja(d, x, y1, w - 40, h, i + 1, t, ls, ac)
        if i < 3:
            flecha(d, x + w - 40 + 6, y1 + h // 2, x + w + gap - 6 + 0, y1 + h // 2)

    # El resultado de la postura: brazos / piernas
    xp = x0 + 3 * (w + gap)
    chips = [("BRAZOS", CIAN), ("PIERNAS", MAGENTA), ("AMBAS", ORO), ("NINGUNA", SUAVE)]
    for i, (txt, col) in enumerate(chips):
        cx = xp + i * 85
        d.rounded_rectangle([cx * E, (y1 + h + 14) * E, (cx + 78) * E, (y1 + h + 42) * E], radius=14 * E, outline=col, width=2 * E)
        d.text(((cx + 39) * E, (y1 + h + 28) * E), txt, font=fuente(11, True), fill=col, anchor="mm")

    # Puente OSC: una banda ancha entre Python y TouchDesigner
    yb = 440
    cxp = xp + (w - 40) // 2
    flecha(d, cxp, y1 + h + 46, cxp, yb - 2, MORADO)
    d.rounded_rectangle([60 * E, yb * E, 1700 * E, (yb + 50) * E], radius=25 * E, fill=(30, 22, 56), outline=MORADO, width=2 * E)
    d.text((880 * E, (yb + 25) * E),
           "OSC por UDP  (127.0.0.1 : 7000)  -  posiciones, velocidades y postura de 12 articulaciones",
           font=fuente(17, True), fill=(196, 181, 253), anchor="mm")
    flecha(d, 380, yb + 50, 380, yb + 106, MORADO)

    # Fila 2: TouchDesigner
    d.text((60 * E, 535 * E), "TOUCHDESIGNER  ·  dibuja la luz", font=fuente(15, True), fill=MORADO, anchor="lm")
    y2 = 558
    datos2 = [
        ("OSC In + Filter", ["Recibe posiciones y velocidades", "Suaviza el movimiento", "Recibe arms_open / legs_open"], MORADO),
        ("Script TOP", ["Cinta continua por cada miembro", "Brazos cian · piernas magenta", "Abierto: más gruesa y cálida"], MORADO),
        ("Feedback + Blur", ["Cada cuadro se apaga un poco", "La estela queda como cola de luz", "Fondo negro"], MORADO),
        ("Pantalla secundaria", ["Ventana a pantalla completa", "Solo la luz, sin la cámara", "Es lo que ve el público"], ORO),
    ]
    for i, (t, ls, ac) in enumerate(datos2):
        x = x0 + i * (w + gap)
        caja(d, x, y2, w - 40, h, i + 5, t, ls, ac)
        if i < 3:
            flecha(d, x + w - 40 + 6, y2 + h // 2, x + w + gap - 6, y2 + h // 2)

    # Nota de la interfaz
    d.rounded_rectangle([60 * E, 775 * E, 1740 * E, 830 * E], radius=16 * E, fill=(18, 18, 30), outline=BORDE, width=2 * E)
    d.text((90 * E, 803 * E),
           "Esta interfaz muestra en vivo la cámara con el esqueleto, los FPS, el % de brazos y piernas abiertos y la postura, "
           "y puede apagar la detección o abrir TouchDesigner.",
           font=fuente(16), fill=SUAVE, anchor="lm")

    img = img.resize((W, H), Image.LANCZOS)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    img.save(SALIDA, optimize=True)
    print(f"Diagrama guardado en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
