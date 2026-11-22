/* ShadowDance - interfaz. Habla con servidor.py por /api/... y muestra el video de /video. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// Llama al servidor: sin cuerpo = GET; con cuerpo = POST (JSON). Devuelve la respuesta ya convertida.
async function api(ruta, cuerpo) {
  const r = await fetch(ruta, cuerpo === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) });
  return r.json();
}

// Estado de la pagina: vista actual, ultimo estado del motor, videos, video elegido, resultados, fallos seguidos.
const S = { vista: "inicio", est: {}, videos: [], sel: null, res: null, fallos: 0 };
const POSTURAS = [
  { clave: "brazos", texto: "BRAZOS", color: "var(--cian)" },
  { clave: "piernas", texto: "PIERNAS", color: "var(--magenta)" },
  { clave: "ambas", texto: "AMBAS", color: "var(--oro)" },
  { clave: "ninguna", texto: "NINGUNA", color: "var(--gris)" },
];

/* ------------------------------------------------------------------ barras en vivo ---- */
// Construye el panel de barras (FPS, cuerpo visible, brazos, piernas, ms, postura) y devuelve su funcion de actualizar.
function crearBarras(raiz) {
  raiz.innerHTML = `
  <div class="panel barras">
    <div class="fps-card">
      <div><div class="fps-et">Velocidad</div><div class="fps-grande"><span class="v-fps">0</span><small>FPS</small></div></div>
      <span class="fps-estado">Sin señal</span>
    </div>
    <div class="carril" style="margin-top:14px"><div class="relleno r-fps"></div></div>
    <div class="medidor"><div class="medidor-fila"><span>Cuerpo visible</span><b class="v-vis">0 %</b></div><div class="carril"><div class="relleno gris r-vis"></div></div></div>
    <div class="medidor"><div class="medidor-fila"><span>Brazos abiertos</span><b class="v-bra">0 %</b></div><div class="carril"><div class="relleno cian r-bra"></div></div></div>
    <div class="medidor"><div class="medidor-fila"><span>Piernas abiertas</span><b class="v-pie">0 %</b></div><div class="carril"><div class="relleno magenta r-pie"></div></div></div>
    <div class="medidor"><div class="medidor-fila"><span>Tiempo por cuadro</span><b class="v-ms">0 ms</b></div><div class="carril"><div class="relleno verde r-ms"></div></div></div>
    <div class="postura-titulo">Postura detectada</div>
    <div class="chips">${POSTURAS.map(p => `<div class="chip" data-p="${p.clave}" style="--c:${p.color}">${p.texto}</div>`).join("")}</div>
    <div class="estado-linea"></div>
  </div>`;
  const q = s => $(s, raiz);
  const panel = q(".barras");
  return function actualizar(e) {
    const act = !!e.activo, ver = act && e.detectar;
    panel.classList.toggle("apagado", !act || !e.detectar);
    const fps = act ? (e.fps || 0) : 0;
    q(".v-fps").textContent = Math.round(fps);
    q(".r-fps").style.width = Math.min(100, fps / 35 * 100) + "%";
    const est = q(".fps-estado");
    const nivel = !act ? ["", "Sin señal"] : fps >= 24 ? ["bien", "Fluido"] : fps >= 15 ? ["regular", "Aceptable"] : ["mal", "Lento"];
    est.className = "fps-estado " + nivel[0]; est.textContent = nivel[1];
    q(".r-fps").className = "relleno r-fps " + (nivel[0] === "mal" ? "rojo" : nivel[0] === "regular" ? "amarillo" : "");

    const vis = ver && e.detectado ? e.visibilidad || 0 : 0;
    q(".v-vis").textContent = Math.round(vis) + " %"; q(".r-vis").style.width = vis + "%";
    const bra = ver && e.detectado && e.brazos != null ? e.brazos : 0;
    const pie = ver && e.detectado && e.piernas != null ? e.piernas : 0;
    q(".v-bra").textContent = Math.round(bra) + " %"; q(".r-bra").style.width = bra + "%"; q(".r-bra").classList.toggle("abierto", bra >= 50);
    q(".v-pie").textContent = Math.round(pie) + " %"; q(".r-pie").style.width = pie + "%"; q(".r-pie").classList.toggle("abierto", pie >= 50);
    const ms = act ? e.ms_cuadro || 0 : 0;
    q(".v-ms").textContent = ms.toFixed(0) + " ms"; q(".r-ms").style.width = Math.min(100, ms / 80 * 100) + "%";
    q(".r-ms").className = "relleno r-ms " + (ms > 50 ? "rojo" : ms > 36 ? "amarillo" : "verde");

    $$(".chip", raiz).forEach(c => c.classList.toggle("on", ver && e.detectado && e.clase === c.dataset.p));
    q(".estado-linea").textContent = textoEstado(e);
  };
}

// Texto corto de la pastilla de estado segun lo que haga el motor.
function textoEstado(e) {
  if (!e.activo) return e.cargando ? "Cargando el modelo…" : "";
  if (!e.detectar) return "Detección apagada: solo se ve la imagen";
  if (!e.detectado) return "Buscando a una persona…";
  if (!e.clasificador) return "Falta modelos\\postura.joblib (la postura no se clasifica)";
  return e.clase ? "Postura: " + e.clase.toUpperCase() : "";
}

const barras = { camara: null, video: null };

/* ------------------------------------------------------------------------ navegacion ---- */
// Cambia de pantalla (inicio, ejecutar, pruebas, como funciona).
function ir(vista) {
  S.vista = vista;
  $$(".vista").forEach(v => v.classList.toggle("activa", v.id === "v-" + vista));
  $$(".nav-item").forEach(b => b.classList.toggle("activo", b.dataset.vista === vista));
  if (vista === "pruebas") { cargarVideos(); cargarResultados(); }
  $(".contenido").scrollTo({ top: 0 });
}
$$(".nav-item").forEach(b => b.addEventListener("click", () => ir(b.dataset.vista)));
$$("[data-ir]").forEach(b => b.addEventListener("click", () => ir(b.dataset.ir)));

/* ----------------------------------------------------------------------------- visor ---- */
const VISORES = [["ejecutar", "camara"], ["pruebas", "video"]];
// Vuelve a conectar el <img> al /video (o lo deja en blanco) cuando se inicia o se detiene algo.
function reiniciarVisor(vista) {
  const img = $(".visor-img", $("#visor-" + vista));
  img.removeAttribute("src"); delete img.dataset.on;
}
$$(".visor-img").forEach(img => img.addEventListener("error", () => { delete img.dataset.on; }));

// Muestra el video en vivo solo en la pantalla que corresponde al modo actual (camara o video).
function actualizarVisores(e) {
  for (const [vista, modo] of VISORES) {
    const visor = $("#visor-" + vista), img = $(".visor-img", visor);
    const quiere = S.vista === vista && e.modo === modo && (e.activo || e.cargando);
    if (quiere) {
      if (!img.dataset.on) { img.src = "/video?" + Date.now(); img.dataset.on = "1"; }
    } else if (img.dataset.on) { img.removeAttribute("src"); delete img.dataset.on; }
    const mostrar = quiere && (e.cuadros || 0) > 0;
    visor.classList.toggle("con-video", mostrar);
    $(".fps-num", visor).textContent = Math.round(e.fps || 0);
    const aviso = $(".visor-aviso", visor);
    const t = mostrar ? textoEstado(e) : "";
    aviso.textContent = t; aviso.classList.toggle("ver", !!t && e.activo);
  }
}

/* -------------------------------------------------------------------------- estado ---- */
// Pone texto y color a una pastilla de estado.
function pastilla(id, texto, clase) {
  const p = $("#" + id); p.className = "pastilla " + clase; $("span", p).textContent = texto;
}

// Dibuja en pantalla todo lo que dice el estado: pastillas, botones, mensajes y barras.
function pintar(e) {
  S.est = e;
  const cam = e.modo === "camara", vid = e.modo === "video";
  const cargando = !!e.cargando;

  // Pastillas de estado
  const vivoCam = cam && e.activo, vivoVid = vid && e.activo;
  pastilla("pill-ejecutar", cam && cargando ? "Cargando…" : vivoCam ? (e.detectar ? "En vivo" : "En vivo · sin detección") : "Detenido",
    cam && cargando ? "carga" : vivoCam ? "vivo" : "");
  pastilla("pill-pruebas", vid && cargando ? "Cargando…" : vivoVid ? (e.pausado ? "En pausa" : "Reproduciendo") : "Detenido",
    vid && cargando ? "carga" : vivoVid && !e.pausado ? "vivo" : "");

  // Boton principal de Ejecutar
  const bi = $("#btn-iniciar"), corriendo = cam && (e.activo || cargando);
  bi.classList.toggle("detener", corriendo);
  bi.innerHTML = corriendo ? "■&nbsp; Detener" : "▶&nbsp; Iniciar ShadowDance";
  bi.disabled = false;

  // Pruebas: pausa
  $("#btn-pausa").innerHTML = vid && e.pausado ? "▶&nbsp; Continuar" : "▮▮&nbsp; Pausa";
  $("#nombre-video").textContent = (e.fuente || (S.sel ? S.sel.nombre : "VIDEO")).toUpperCase();

  // Mensajes
  const mensaje = (id, activo) => {
    const el = $("#" + id);
    const err = activo && e.error;
    el.className = "mensaje" + (err ? " error" : "");
    el.textContent = err ? e.error : (activo && e.mensaje) || "";
  };
  mensaje("msg-ejecutar", cam || (!e.activo && e.error && e.modo === "camara"));
  mensaje("msg-pruebas", vid);

  if (cam) barras.camara(e); else barras.camara({ activo: false, detectar: true });
  if (vid) barras.video(e); else barras.video({ activo: false, detectar: true });
  actualizarVisores(e);
}

// Pregunta el estado cada ~0.3 s. Ademas sirve de "latido": si la ventana se cierra, el servidor se apaga solo.
async function sondear() {
  try {
    const e = await api("/api/estado");
    e.cargando = e.modo && !e.activo && !e.terminado && !e.error;
    S.fallos = 0; pintar(e);
  } catch (_) {
    if (++S.fallos >= 4) $("#cierre").classList.add("ver");
  }
}

/* --------------------------------------------------------------------------- ejecutar ---- */
$("#btn-iniciar").addEventListener("click", async () => {
  const e = S.est;
  if (e.modo === "camara" && (e.activo || e.cargando)) { await api("/api/detener", {}); reiniciarVisor("ejecutar"); return; }
  const b = $("#btn-iniciar"); b.disabled = true; b.innerHTML = "Cargando…";
  reiniciarVisor("ejecutar");
  await api("/api/iniciar", {
    osc: true, detectar: $("#sw-deteccion").checked,
    clahe: $("#sw-clahe").checked, camara: parseInt($("#in-camara").value, 10) || 0,
  });
});
const ajustar = datos => api("/api/ajustes", datos);
$("#sw-deteccion").addEventListener("change", ev => S.est.modo === "camara" && ajustar({ detectar: ev.target.checked }));
$("#sw-clahe").addEventListener("change", ev => S.est.modo === "camara" && ajustar({ clahe: ev.target.checked }));

/* ---------------------------------------------------------------------------- pruebas ---- */
// Pide la lista de videos de prueba y arma las tarjetas.
async function cargarVideos() {
  S.videos = await api("/api/videos");
  const lista = $("#lista-videos");
  if (!S.videos.length) {
    lista.innerHTML = `<div class="vacio">No hay videos todavía. Copia los videos a <b>INTERFAZ\\videos</b> (o a <b>clips</b>) y pulsa ↻.</div>`;
    S.sel = null; return;
  }
  if (!S.sel || !S.videos.find(v => v.id === S.sel.id)) S.sel = S.videos[0];
  lista.innerHTML = S.videos.map(v => `
    <button class="video-item ${S.sel.id === v.id ? "sel" : ""}" data-id="${v.id}">
      <b>${v.nombre}</b><span>${v.duracion ? v.duracion + " s · " : ""}${v.resolucion || ""} · ${v.carpeta}</span>
    </button>`).join("");
  $$(".video-item", lista).forEach(b => b.addEventListener("click", () => {
    S.sel = S.videos.find(v => v.id === b.dataset.id);
    $$(".video-item", lista).forEach(x => x.classList.toggle("sel", x === b));
    renderMedicion();
  }));
  renderMedicion();
}
$("#btn-recargar").addEventListener("click", () => { cargarVideos(); cargarResultados(); });

// Lee results\\*.csv a traves del servidor y llena las dos tablas.
async function cargarResultados() { S.res = await api("/api/resultados"); renderMedicion(); renderClasificador(); }

// Tabla de mediciones del sistema (deteccion, ms por cuadro, FPS por condicion).
function renderMedicion() {
  const caja = $("#tabla-medicion");
  if (!S.res) return;
  const filas = S.sel ? (S.res.sistema || []).filter(r => r.video === S.sel.nombre) : [];
  if (!filas.length) {
    caja.innerHTML = `<div class="vacio">Todavía no hay mediciones de este video.<br>Se generan con <b>python src\\evaluar_sistema.py --videos …</b></div>`;
    return;
  }
  caja.innerHTML = `<table><tr><th>Condición</th><th class="num">Detección</th><th class="num">ms</th><th class="num">FPS</th></tr>` +
    filas.map(r => `<tr><td>${r.condicion}</td><td class="num">${(+r.deteccion_pct).toFixed(0)} %</td>
      <td class="num">${(+r.ms_cuadro_media).toFixed(0)}</td><td class="num">${(+r.fps_posibles).toFixed(0)}</td></tr>`).join("") + `</table>`;
}

// Tabla de acierto del clasificador por esquema de evaluacion.
function renderClasificador() {
  const caja = $("#tabla-clasificador"), panel = $("#panel-clasificador");
  const filas = ((S.res && S.res.clasificador) || []).filter(r => r.pregunta === "las dos a la vez");
  panel.style.display = filas.length ? "" : "none";
  if (!filas.length) return;
  const modelos = [...new Set(filas.map(r => r.modelo))];
  const esquemas = [...new Set(filas.map(r => r.esquema))];
  caja.innerHTML = `<table><tr><th>Prueba</th><th class="num">Muestras</th>${modelos.map(m => `<th class="num">${m}</th>`).join("")}</tr>` +
    esquemas.map(es => {
      const sub = filas.filter(r => r.esquema === es);
      const vals = modelos.map(m => { const r = sub.find(x => x.modelo === m); return r ? +r.accuracy : null; });
      const mx = Math.max(...vals.filter(v => v != null));
      return `<tr><td>${es}</td><td class="num">${sub[0].n}</td>` +
        vals.map(v => `<td class="num ${v === mx ? "mejor" : ""}">${v == null ? "–" : (v * 100).toFixed(1) + " %"}</td>`).join("") + `</tr>`;
    }).join("") + `</table>`;
}

$("#btn-reproducir").addEventListener("click", async () => {
  if (!S.sel) { $("#msg-pruebas").textContent = "No hay videos para reproducir."; return; }
  reiniciarVisor("pruebas");
  const osc = $("#sw-p-osc").checked;
  await api("/api/reproducir", {
    id: S.sel.id, osc, detectar: $("#sw-p-deteccion").checked,
    clahe: $("#sw-p-clahe").checked, bucle: $("#sw-p-bucle").checked,
  });
});
$("#btn-pausa").addEventListener("click", () => S.est.modo === "video" && ajustar({ pausado: !S.est.pausado }));
$("#btn-reiniciar").addEventListener("click", () => S.est.modo === "video" && ajustar({ reiniciar: true }));
$("#btn-parar").addEventListener("click", async () => { await api("/api/detener", {}); reiniciarVisor("pruebas"); });
for (const [id, clave] of [["sw-p-deteccion", "detectar"], ["sw-p-clahe", "clahe"], ["sw-p-bucle", "bucle"], ["sw-p-osc", "osc"]]) {
  $("#" + id).addEventListener("change", ev => S.est.modo === "video" && ajustar({ [clave]: ev.target.checked }));
}

/* ------------------------------------------------------------------------------ salir ---- */
$("#btn-salir").addEventListener("click", async () => {
  try { await api("/api/salir", {}); } catch (_) { /* el servidor ya se cerro */ }
  $("#cierre").classList.add("ver");
  setTimeout(() => window.close(), 400);
});

/* ------------------------------------------------------------------------------ inicio ---- */
barras.camara = crearBarras($("#barras-ejecutar"));
barras.video = crearBarras($("#barras-pruebas"));
barras.camara({ activo: false, detectar: true });
barras.video({ activo: false, detectar: true });
api("/api/config").then(c => { if (c && c.camara != null) $("#in-camara").value = c.camara; }).catch(() => {});
setInterval(sondear, 150);
sondear();
