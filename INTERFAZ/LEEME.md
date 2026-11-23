# Interfaz de ShadowDance

Interfaz local para **ejecutar** ShadowDance, **probar** con videos grabados y **explicar** cómo funciona.
Solo *usa* el código de `src\` y de `touchdesigner\`; no lo modifica.

## Cómo abrirla

Desde la raíz del proyecto, con el entorno `.venv` activo:

```
python INTERFAZ\iniciar.py
```

Se abre una ventana propia (Edge en modo aplicación; si no hay Edge, el navegador). Se cierra con el botón **Salir**
o al cerrar la ventana. No hay que instalar nada extra: usa lo mismo que el resto del proyecto (opencv, mediapipe, python-osc).

> Cierra `pose_osc.py` antes de abrirla: la interfaz usa la cámara y manda el OSC ella misma.

## Pantallas

| Pantalla | Para qué sirve |
|---|---|
| **Inicio** | Menú principal: título, creadores, materia y accesos. |
| **Ejecutar** | Enciende la cámara con el esqueleto de brazos y piernas, FPS, % de cuerpo visible, % de brazos y piernas abiertos, tiempo por cuadro y la postura (BRAZOS / PIERNAS / AMBAS / NINGUNA). |
| **Pruebas** | Reproduce videos grabados con la detección encima y muestra sus mediciones (`results\sistema_metricas.csv`). |
| **Cómo funciona** | Diagrama de todo el sistema (`assets\como_funciona.png`). |

En **Ejecutar**:
- **Iniciar ShadowDance**: arranca la cámara, la detección y el envío OSC. El mismo botón pasa a **Detener**.
- **Detección** (interruptor): apaga o prende la detección sin cerrar la cámara. Apagada, no se manda nada a TouchDesigner.
- **Mejora de luz (CLAHE)**: ecualización de histograma antes de detectar; ayuda con poca luz.
- **Cámara**: número de la cámara (la PS3 Eye del equipo es la 1; `python src\listar_camaras.py` ayuda a encontrarla).

En **Pruebas**: elige un video, **Reproducir**, y usa Pausa, Reiniciar, Repetir y los interruptores.
**Enviar a TouchDesigner** manda el video por OSC como si fuera la cámara, para ver la estela.

## Videos de prueba

La interfaz busca videos (`.mp4`, `.mov`, `.avi`, `.mkv`) en las carpetas de `carpetas_videos` de `config.json`:
`INTERFAZ\videos`, `INTERFAZ\videos\usuarios`, `clips` y `lab_deteccion`. Pulsa el botón de recargar tras agregar uno.
Los videos **no se suben a GitHub** (`*.mp4` está en `.gitignore`).

- Grabar los 4 videos de un usuario: `python src\grabar_video.py --usuario ana --camara 1 --sin-mjpg --ancho 640 --alto 480`
  (quedan en `INTERFAZ\videos\usuarios\`).
- Medirlos: `python src\evaluar_sistema.py --videos ...` (crea `results\sistema_metricas.csv`).
- Videos de celular (4K): primero `python src\optimizar_video.py video.mp4`, que crea una copia liviana `_lite.mp4`.

## Rendimiento

- **Modo adaptativo (automático):** si el computador no alcanza a detectar en cada cuadro, detecta 1 de cada 2 y
  reutiliza la última pose en los cuadros intermedios. El video y el dibujo siguen a velocidad completa.
- **Modelo ligero (opcional):** descarga `pose_landmarker_lite.task` a `src\` y agrega `"modelo": "lite"` a `config.json`
  (más rápido, algo menos preciso; comprueba la postura antes de usarlo).
- **Diagnóstico:** `python src\diagnostico_rendimiento.py --video ...` muestra cuánto tarda cada etapa.
- Tiempo por cuadro = trabajo propio (sin esperar a la cámara). Con poca luz la cámara entrega pocos cuadros por segundo
  y los FPS bajan aunque el programa vaya sobrado.

## TouchDesigner (pantalla secundaria)

La interfaz **no abre TouchDesigner**: abre `touchdesigner\shadowdance3.toe` a mano **antes** de iniciar.
Solo manda los datos por OSC (127.0.0.1:7000). Para la pantalla negra, en TouchDesigner pulsa **F1**
(abre la ventana de salida) y **Esc** para cerrarla.

## Configuración (`config.json`)

| Clave | Significado |
|---|---|
| `puerto` | Puerto local de la interfaz (8765). |
| `camara`, `ancho`, `alto` | Cámara y resolución (640×480). |
| `sin_mjpg`, `backend` | Opciones de la cámara en Windows (DirectShow, sin MJPG). |
| `osc_ip` | A dónde se envía el OSC (127.0.0.1). |
| `carpetas_videos` | Carpetas donde se buscan los videos de la pestaña Pruebas. |
| `modelo` | Opcional: `"lite"` para el modelo de pose ligero. |

## Qué hay en la carpeta

| Archivo | Función |
|---|---|
| `iniciar.py` | Arranca el servidor local y abre la ventana. |
| `servidor.py` | Página + API local (video MJPEG, estado, controles). |
| `motor.py` | Cámara o video → MediaPipe → clasificador → OSC, en un hilo aparte. |
| `config.json` | Opciones (ver arriba). |
| `generar_diagrama.py` | Dibuja `assets\como_funciona.png`. |
| `web\` | Página (HTML, CSS, JS). |
| `videos\` | Videos de prueba (no se suben a GitHub). |
