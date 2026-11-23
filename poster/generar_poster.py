"""
generar_poster.py - Poster ShadowDance (A1 vertical, 594 x 841 mm) en SVG editable.
Colores UAO (manual de identidad): rojo Pantone 1795 C y negro. Se abre en Affinity: Archivo > Abrir.
Uso:  python generar_poster.py   (lee las figuras de ../poster_figuras o de figuras/)
"""
import base64
import os

# ---------------------------------------------------------------- rutas, colores y fuente ---
AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = next(p for p in (os.path.join(AQUI, "figuras"), os.path.join(AQUI, "..", "poster_figuras")) if os.path.isdir(p))
ROJO, NEGRO, GRIS, CLARO, BORDE = "#D22630", "#111111", "#8C8C8C", "#F2F2F2", "#D9D9D9"
ORO, CIAN, MAGENTA, NARANJA = "#FBBF24", "#22D3EE", "#F472B6", "#FB923C"
FUENTE = "Arial, Helvetica, sans-serif"
o = []          # aqui se van acumulando los fragmentos de SVG


# ---------------------------------------------------------------- funciones para dibujar ---
def add(s):
    """Agrega un fragmento de SVG al poster."""
    o.append(s)


def rect(x, y, w, h, fill, r=0, stroke=None, sw=0.6, dash=None, op=1):
    """Rectangulo (opcional: esquinas redondas r, borde, linea punteada y opacidad)."""
    a = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" fill-opacity="{op}"'
    if stroke: a += f' stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else "")
    add(a + "/>")


def txt(x, y, lineas, size, fill=NEGRO, peso="normal", anchor="start", interl=1.25, estilo=""):
    """Texto de una o varias lineas (lista); interl = separacion entre lineas como multiplo del tamano."""
    if isinstance(lineas, str):
        lineas = [lineas]
    t = f'<text x="{x}" y="{y}" font-family="{FUENTE}" font-size="{size}" font-weight="{peso}" fill="{fill}" text-anchor="{anchor}" {estilo}>'
    for i, l in enumerate(lineas):
        l = l.replace("&", "&amp;")
        t += f'<tspan x="{x}" dy="{0 if i == 0 else round(size * interl, 2)}">{l}</tspan>'
    add(t + "</text>")


def img(nombre, x, y, w, h):
    """Inserta un PNG de poster/figuras dentro del SVG (incrustado en base64, para que el archivo sea autonomo)."""
    b = base64.b64encode(open(os.path.join(FIG, nombre), "rb").read()).decode()
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{b}"/>')


def titulo(x, y, s):
    """Titulo de seccion: barrita roja + texto en negrita."""
    rect(x, y - 11, 5, 14, ROJO)
    txt(x + 10, y, s, 13, NEGRO, "bold")

add('<svg xmlns="http://www.w3.org/2000/svg" width="594mm" height="841mm" viewBox="0 0 594 841">')
rect(0, 0, 594, 841, "#FFFFFF")

# ---------------------------------------------------------------- encabezado
rect(0, 0, 594, 105, NEGRO)
rect(0, 105, 594, 5, ROJO)
txt(20, 58, "ShadowDance", 40, "#FFFFFF", "bold")
txt(20, 76, "Tu cuerpo dibuja la luz: postura en tiempo real", 10.5, "#E6E6E6")
txt(20, 91, "Sebastián Meneses Molina · Juan Camilo Trujillo Vargas · Fanier Andrés Candelo Triana", 8, "#BDBDBD")
rect(484, 16, 90, 30, "none", 3, "#8C8C8C", 0.5, "2 2")
txt(529, 33.5, "Logo UAO", 8, "#8C8C8C", anchor="middle")
txt(574, 64, ["Universidad Autónoma", "de Occidente · Cali"], 8.5, "#FFFFFF", "bold", "end")
txt(574, 87, ["Procesamiento Digital", "de Imágenes · 2026"], 8, "#BDBDBD", anchor="end", interl=1.2)

# ---------------------------------------------------------------- A: qué es + ilustración
rect(20, 124, 320, 170, ROJO, 8)
txt(36, 148, "¿Qué es?", 15, "#FFFFFF", "bold")
txt(36, 165, ["Una cámara ve tu cuerpo. Un modelo de",
              "visión detecta tu pose y un clasificador",
              "reconoce si abres brazos, piernas, ambos",
              "o ninguno. TouchDesigner lo convierte",
              "en estelas de luz, en vivo."], 10.5, "#FFFFFF", interl=1.3)
for i, (t, w) in enumerate((("NINGUNA", 58), ("BRAZOS", 54), ("PIERNAS", 58), ("AMBAS", 50))):
    x = 36 + sum(p + 5 for p in (58, 54, 58, 50)[:i])
    rect(x, 262, w, 16, "none", 8, "#FFFFFF", 0.8)
    txt(x + w / 2, 272.8, t, 6.5, "#FFFFFF", "bold", "middle")

rect(354, 124, 220, 170, NEGRO, 8)
cx = 464          # centro horizontal de la ilustracion


def trazo(d, col, w, op):
    """Trazo de luz curvo (se dibuja varias veces, de grueso y tenue a fino y brillante, para dar efecto de estela)."""
    add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w}" stroke-opacity="{op}" stroke-linecap="round"/>')
# estelas de luz de los cuatro miembros
for w, op in ((9, .12), (6, .22), (3.4, .5), (1.4, .95)):
    trazo("M418,166 C396,160 380,178 372,206 C368,222 372,238 380,252", CIAN, w, op)
    trazo("M510,166 C532,160 548,178 556,206 C560,222 556,238 548,252", MAGENTA, w, op)
    trazo("M444,252 C436,262 424,272 408,276", NARANJA, w, op)
    trazo("M484,252 C492,262 504,272 520,276", ORO, w, op)
add(f'<circle cx="{cx}" cy="156" r="9" fill="none" stroke="#FFFFFF" stroke-width="2.6"/>')
add(f'<line x1="{cx}" y1="165" x2="{cx}" y2="222" stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round" stroke-opacity=".35"/>')
for (x1, y1, x2, y2, c) in ((cx, 178, 418, 166, CIAN), (cx, 178, 510, 166, MAGENTA), (cx, 222, 444, 252, NARANJA), (cx, 222, 484, 252, ORO)):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="2.6" stroke-linecap="round"/>')
for (x, y) in ((418, 166), (510, 166), (444, 252), (484, 252)):
    add(f'<circle cx="{x}" cy="{y}" r="2.6" fill="#FFFFFF"/>')
txt(464, 288, "Pantalla secundaria (ilustración)", 5.5, "#8C8C8C", anchor="middle")

# ---------------------------------------------------------------- B: cómo funciona
titulo(20, 316, "Cómo funciona")
pasos = [("Cámara", ["640×480", "~30 FPS"]), ("MediaPipe", ["33 puntos", "de pose"]), ("Clasificador", ["brazos y", "piernas"]),
         ("OSC", ["datos por", "UDP"]), ("TouchDesigner", ["estela", "de luz"])]
for i, (t, d) in enumerate(pasos):
    x = 20 + i * 113.5
    rect(x, 326, 100, 88, CLARO, 8)
    add(f'<circle cx="{x + 17}" cy="{344}" r="9" fill="{ROJO}"/>')
    txt(x + 17, 347.2, str(i + 1), 9, "#FFFFFF", "bold", "middle")
    txt(x + 8, 372, t, 9.5, NEGRO, "bold")
    txt(x + 8, 388, d, 8, "#555555", interl=1.25)
    if i < 4:
        add(f'<polygon points="{x + 102},{413 - 44} {x + 102},{413 - 32} {x + 111},{413 - 38}" fill="{ROJO}"/>')

# ---------------------------------------------------------------- C: resultados
titulo(20, 442, "Resultados")
for x, f, c1, c2 in ((20, "fig_acierto_4clases.png", "Con personas nuevas, las 4 posturas", "se aciertan ~62 % (azar: 25 %)"),
                     (304, "fig_clahe_oscuridad.png", "Con solo 4 % de luz, CLAHE sube la detección", "de 0 % a 83 % (en brillo 6 %: de 50 % a 99 %)")):
    rect(x, 452, 270, 190, "#FFFFFF", 8, BORDE, 0.8)
    img(f, x + 6, 455, 258, 152)
    txt(x + 135, 620, c1, 8, NEGRO, "bold", "middle")
    txt(x + 135, 630, c2, 8, NEGRO, "bold", "middle")
rect(20, 654, 554, 54, ROJO, 8)
for i, (n, l) in enumerate((("83 %", "detección con 4 % de luz + CLAHE"), ("≈ 40 FPS", "capacidad de proceso (cámara: 30)"), ("~75 %", "acierto por pregunta (brazos/piernas)"))):
    cxx = 20 + 92.3 + i * 184.7
    txt(cxx, 681, n, 22, "#FFFFFF", "bold", "middle")
    txt(cxx, 696, l, 7, "#FFFFFF", anchor="middle")

# ---------------------------------------------------------------- D: logros / límites / mejoras
for i, (t, col, b) in enumerate((("Logros", ROJO, ["• Sistema completo en tiempo real", "• CLAHE recupera la detección sin luz"]),
                                 ("Límites", NEGRO, ["• Solo 2 personas en las capturas", "• Etiquetas por categoría, no manuales"]),
                                 ("Mejoras", "#555555", ["• Más personas, fondos y luces", "• Usar secuencias de cuadros"]))):
    x = 20 + i * 187.7
    rect(x, 720, 178.6, 68, CLARO, 8)
    rect(x, 720, 5, 68, col, 2)
    txt(x + 13, 735, t, 10, col, "bold")
    txt(x + 13, 748, b, 7.5, "#333333", interl=1.4)

# ---------------------------------------------------------------- pie
rect(0, 799, 594, 42, NEGRO)
rect(0, 796, 594, 3, ROJO)
txt(20, 810, ["Referencias: Lugaresi et al. (2019), MediaPipe, arXiv:1906.08172 · Bazarevsky et al. (2020), BlazePose, arXiv:2006.10204",
              "Zuiderveld (1994), CLAHE, Graphics Gems IV · Utterback y Achituv (1999), Text Rain",
              "Código y datos: github.com/…  (completar)"], 5.2, "#E6E6E6", interl=1.5)
rect(536, 803, 34, 34, "#FFFFFF", 2)
txt(553, 823, "QR", 8, GRIS, "bold", "middle")
add("</svg>")
open(os.path.join(AQUI, "ShadowDance_poster.svg"), "w", encoding="utf-8").write("\n".join(o))
print("ok")
