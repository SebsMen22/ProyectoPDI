# Interfaz de ShadowDance

Una interfaz sencilla para **ejecutar** ShadowDance, **probar** con videos grabados y **explicar** como funciona.
Esta carpeta es nueva: **no modifica nada de `src\` ni de `touchdesigner\`**, solo los usa.

## Como abrirla

Desde la raiz del proyecto, con el entorno `.venv` activo:

```
python INTERFAZ\iniciar.py
```

Se abre una ventana propia (Edge en modo aplicacion; si no hay Edge, tu navegador). Se cierra con el boton **Salir**
o al cerrar la ventana. No hay que instalar nada nuevo: usa lo mismo que ya tienes (opencv, mediapipe, python-osc).

> Cierra `pose_osc.py` antes de abrirla: la interfaz usa la camara y manda el OSC ella misma.

## Pantallas

| Pantalla | Para que sirve |
|---|---|
| **Inicio** | Menu principal: titulo, creadores, materia y accesos. |
| **Ejecutar** | Enciende la camara con el esqueleto encima, FPS, % de cuerpo visible, % de brazos y piernas abiertos y la postura (BRAZOS / PIERNAS / AMBAS / NINGUNA). |
| **Pruebas** | Reproduce los videos grabados con la deteccion encima, y muestra sus mediciones (`results\sistema_metricas.csv`) y el acierto del clasificador. |
| **Como funciona** | Diagrama de todo el sistema (`assets\como_funciona.png`). |

En **Ejecutar**:
- **Iniciar ShadowDance**: arranca la camara, la deteccion y el envio OSC a TouchDesigner. El mismo boton pasa a **Detener**.
- **Deteccion** (interruptor): apaga o prende la deteccion sin cerrar la camara. Apagada, no se manda nada a TouchDesigner.
- **Mejora de luz (CLAHE)**: ecualizacion de histograma antes de detectar; ayuda con poca luz.
- **Camara**: el numero de la camara (tu PS3 era la 1).

En **Pruebas**: elige un video, **Reproducir**, y usa Pausa, Reiniciar, Repetir y los interruptores.
**Enviar a TouchDesigner** manda el video por OSC como si fuera la camara, para ver la estela en TouchDesigner.

## Los videos de prueba

Copia tus videos a `INTERFAZ\videos\` (tambien se buscan en `clips\` y `lab_deteccion\`) y pulsa el boton de recargar.
Las mediciones de cada video salen de `python src\evaluar_sistema.py --videos ...` (ese script crea `results\sistema_metricas.csv`).

## TouchDesigner (pantalla secundaria)

La interfaz **no abre TouchDesigner**: abre `touchdesigner\shadowdance3.toe` a mano **antes** de iniciar.
La interfaz solo manda los datos por OSC (127.0.0.1:7000). Para la pantalla negra, en TouchDesigner pulsa **F1**
(abre la ventana de salida, `window1`) y **Esc** para cerrarla.

## Que hay en la carpeta

| Archivo | Funcion |
|---|---|
| `iniciar.py` | Arranca el servidor local y abre la ventana. |
| `servidor.py` | Pagina + API local (video MJPEG, estado, controles). |
| `motor.py` | Camara/video -> MediaPipe -> clasificador -> OSC, en un hilo aparte. |
| `config.json` | Puerto, camara, resolucion, carpetas de videos. |
| `generar_diagrama.py` | Dibuja `assets\como_funciona.png`. |
| `web\` | Pagina (HTML, CSS, JS). |
