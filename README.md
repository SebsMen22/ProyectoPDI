# Shadow Dance

Sistema interactivo de visión computacional que detecta el cuerpo de una persona con una cámara y convierte su movimiento en una **estela de colores** que se muestra en tiempo real en una pantalla secundaria.

- **Curso:** Procesamiento Digital de Imágenes — Universidad Autónoma de Occidente (UAO), Cali, Colombia
- **Docente:** Nicolas Llanos Neuta
- **Integrantes:** Juan Camilo Trujillo Vargas, Sebastián Meneses Molina, Fanier Andrés Candelo Triana
- **Macroproyecto:** Análisis de la Expresión Humana en el Arte y la Cultura

> Estado: en desarrollo. Las secciones marcadas con *(pendiente)* se completan a medida que avanza el proyecto.

---

## 1. Descripción del proyecto

**Problemática.** En la danza, el movimiento es una expresión efímera: una vez ejecutado, desaparece en el tiempo y el espacio. Falta un medio interactivo que extienda y haga visible la trayectoria del movimiento.

**Objetivo general.** Desarrollar un sistema interactivo basado en procesamiento digital de imágenes y visión computacional que potencie el atractivo visual del baile capturando y exaltando el movimiento humano en tiempo real.

**Objetivos específicos** *(el tercero se ajustó a la estela; confirmar con el grupo)*:
1. Detectar, seguir y extraer parámetros de movimiento de una persona en tiempo real.
2. Integrar un modelo de estimación de pose para identificar articulaciones y posturas clave.
3. Traducir las métricas de movimiento (posición, velocidad, amplitud, postura) en parámetros visuales que generen una estela de colores sincronizada con el cuerpo.

**Solución propuesta.** Un programa en Python detecta la pose y calcula posición y velocidad de cada articulación; envía esos datos por OSC a TouchDesigner, que dibuja la estela y la proyecta en una pantalla secundaria. Un panel de control permite configurar y monitorear el sistema.

## 2. Arquitectura del sistema

Cámara → **Python** (MediaPipe Pose → métricas de movimiento → mensajes OSC) → **TouchDesigner** (OSC In → suavizado → pincel → feedback → ventana) → pantalla secundaria.

| Bloque | Función | Entrada | Salida |
|---|---|---|---|
| Captura | Obtener video de la cámara o de un archivo | Cuerpo en movimiento | Cuadros BGR |
| Pose | Detectar 33 puntos del cuerpo (MediaPipe) | Cuadro RGB | Coordenadas normalizadas + visibilidad |
| Métricas | Suavizar y calcular velocidad por articulación | Coordenadas | x, y, velocidad |
| Clasificación de postura *(pendiente)* | Clasificar la postura a partir de los puntos | x, y de articulaciones | Clase de postura |
| Envío OSC | Transmitir los datos a TouchDesigner | x, y, velocidad | Mensajes UDP (puerto 7000) |
| Estela (TouchDesigner) | Dibujar y desvanecer el rastro de color | Canales OSC | Imagen |
| Pantalla secundaria | Mostrar la estela | Imagen | Visual en vivo |

Diagrama general: *(pendiente: insertar imagen en `docs/`)*. Diseño de referencia: https://canva.link/gopd2nt7fygtfx0

## 3. Metodología

- [ ] Recolección de datos
- [ ] Preparación de datos
- [ ] Selección de técnicas y modelos (benchmark de pose)
- [ ] Implementación y entrenamiento (clasificador de postura)
- [ ] Evaluación (métricas y escenarios)
- [ ] Prototipo interactivo (estela e interfaz)

## 4. Implementación

```
shadow-dance/
├── src/            # pose_osc.py, osc_simulador.py, ...
├── touchdesigner/  # shadowdance.toe
├── notebooks/      # Colab: extracción de puntos, clasificador, evaluación
├── data/           # solo descripción y enlaces (los datos pesados no se suben)
├── docs/           # diagramas, capturas, póster
├── requirements.txt
└── README.md
```

**Instalación y ejecución**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# descargar pose_landmarker_full.task (guía de MediaPipe Pose Landmarker) y ponerlo en src/ junto a pose_osc.py
python src/osc_simulador.py     # prueba con datos simulados (sin cámara)
python src/pose_osc.py          # con webcam; o: python src/pose_osc.py video.mp4
```

En TouchDesigner: abrir `touchdesigner/shadowdance.toe` (OSC In CHOP en el puerto 7000).

## 5. Datos

Conjunto propio de imágenes de posturas en 3 clases (Brazos, PiesAbiertos, PiesCruzados). *(pendiente: cantidad por clase, personas, escenarios, enlace de acceso y autorización de uso de imagen).*

## 6. Interfaz gráfica de usuario

Pantallas diseñadas: Inicio, Configuración, Ejecución, Resultados y Estado de excepción. Prototipo: https://canva.link/79kvrif9gs2wxmj. *(pendiente: capturas, funcionalidades e instrucciones de uso de la versión implementada).*

## 7. Resultados

*(pendiente)* Métricas a reportar: accuracy y F1 por clase con matriz de confusión (partición por persona o sesión), porcentaje de cuadros con persona detectada, FPS y latencia, estabilidad con y sin suavizado, y tabla de robustez por escenario.

## 8. Conclusiones

*(pendiente)* Logros, limitaciones y oportunidades de mejora.

## Licencias y créditos

- TouchDesigner (edición Non-Commercial, uso académico; resolución limitada a 1280×1280).
- Indicar la licencia de cada dependencia usada (por ejemplo, Ultralytics declara AGPL-3.0 si se usa YOLO).
