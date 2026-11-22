"""
iniciar.py  -  Interfaz de Shadow Dance
=======================================
Abre la interfaz. Desde la raiz del proyecto:

    python INTERFAZ\\iniciar.py

Levanta un servidor local (solo en este computador) y abre la pagina en una ventana propia de Edge
(o en tu navegador si no hay Edge). Se cierra sola cuando cierras la ventana, o con el boton Salir.
No modifica nada de src\\ ni de touchdesigner\\: solo los usa.
"""
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from servidor import crear_servidor  # noqa: E402


def cargar_config():
    """Lee INTERFAZ\\config.json (puerto, camara, resolucion, carpetas de videos)."""
    ruta = os.path.join(AQUI, "config.json")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def abrir_ventana(url):
    """Edge en modo 'aplicacion' (sin barra de direcciones). Si no esta, el navegador normal."""
    candidatos = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    for exe in candidatos:
        if os.path.isfile(exe):
            perfil = os.path.join(os.environ.get("TEMP", AQUI), "shadowdance_perfil")
            subprocess.Popen([exe, f"--app={url}", "--window-size=1366,820", f"--user-data-dir={perfil}",
                              "--no-first-run", "--disable-features=Translate"])
            return
    webbrowser.open(url)


def main():
    """Levanta el servidor, abre la ventana y espera hasta que se cierre (boton Salir, Ctrl+C o ventana cerrada)."""
    cfg = cargar_config()
    servidor, estado = crear_servidor(cfg)
    puerto = servidor.server_address[1]
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{puerto}/"
    print(f"Shadow Dance - interfaz en {url}   (Ctrl+C aqui o el boton Salir para cerrar)")
    if "--sin-ventana" not in sys.argv:
        abrir_ventana(url)

    inicio = time.time()
    try:
        while not estado.salir.is_set():
            time.sleep(0.5)
            # Si la ventana se cerro (la pagina dejo de preguntar por el estado), se cierra todo.
            if estado.ultimo_latido and time.time() - estado.ultimo_latido > 20:
                break
            if estado.ultimo_latido is None and time.time() - inicio > 90:
                break
    except KeyboardInterrupt:
        pass
    estado.detener_motor()
    servidor.shutdown()
    print("Interfaz cerrada.")


if __name__ == "__main__":
    main()
