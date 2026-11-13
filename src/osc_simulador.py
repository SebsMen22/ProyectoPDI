"""
osc_simulador.py  -  Shadow Dance
Envía a TouchDesigner el MISMO formato de mensajes OSC que pose_osc.py, pero con movimiento inventado.
Sirve para construir y probar la estela en TouchDesigner sin cámara. Salir: Ctrl + C.
"""
import math
import time

from pythonosc.udp_client import SimpleUDPClient

OSC_IP, OSC_PORT = "127.0.0.1", 7000
NOMBRES = ["lw", "rw", "le", "re", "la", "ra", "lk", "rk"]  # las mismas 8 articulaciones

client = SimpleUDPClient(OSC_IP, OSC_PORT)
t0 = time.time()
print(f"Enviando OSC simulado a {OSC_IP}:{OSC_PORT} ... (Ctrl+C para salir)")
try:
    while True:
        t = time.time() - t0
        client.send_message("/detected", 1.0)
        for i, n in enumerate(NOMBRES):
            fase = i * 0.8                                   # cada articulación va desfasada
            x = 0.5 + 0.30 * math.sin(0.9 * t + fase)        # posición: gira alrededor del centro
            y = 0.5 + 0.30 * math.cos(1.3 * t + fase)
            dx = 0.30 * 0.9 * math.cos(0.9 * t + fase)       # derivada = velocidad en x e y
            dy = -0.30 * 1.3 * math.sin(1.3 * t + fase)
            client.send_message(f"/{n}_x", x)
            client.send_message(f"/{n}_y", y)
            client.send_message(f"/{n}_s", math.hypot(dx, dy))
        time.sleep(1 / 30)                                   # 30 envíos por segundo
except KeyboardInterrupt:
    print("Simulador detenido.")
