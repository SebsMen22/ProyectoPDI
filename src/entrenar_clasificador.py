"""
entrenar_clasificador.py  -  Shadow Dance
=========================================
Entrena y EVALUA el clasificador de postura a partir de data\\keypoints.csv (lo genera extraer_puntos.py).

Dos preguntas de si o no:  abre_brazos  y  abre_piernas  (la clase 'ambas' es si a las dos).
Tres modelos comparados:
    1. Regla de un solo umbral  (arbol de profundidad 1: la mejor medida geometrica con su corte)
    2. Regresion logistica
    3. Random forest
Dos formas de evaluar (las dos evitan que fotos casi iguales caigan en entrenamiento y prueba a la vez):
    A. Validacion cruzada por GRUPOS (sesiones de fotos): 5 pliegues.
    B. Entre PERSONAS: entrena con la persona A y prueba con la B, y al reves.

Salidas (carpeta results\\):
    clasificador_metricas.csv   accuracy y F1 de cada modelo, pregunta y esquema
    clasificador_resumen.md     la misma tabla lista para pegar en el README
    matriz_confusion_*.png      matrices de confusion (clases: brazos / piernas / ambas / ninguna)
    importancia_caracteristicas.csv
Y con --guardar deja el modelo final en modelos\\postura.joblib (no se sube a GitHub).

Uso:
    python src\\entrenar_clasificador.py
    python src\\entrenar_clasificador.py --csv data\\keypoints.csv --salida results --guardar
Necesita:  pip install scikit-learn pandas matplotlib joblib
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postura import NOMBRES, caracteristicas, clase_desde_etiquetas, etiquetas_desde_clase  # noqa: E402

CLASES = ["brazos", "piernas", "ambas", "ninguna"]
OBJETIVOS = ("abre_brazos", "abre_piernas")


def fabricas_de_modelos():
    """{nombre: funcion que crea un modelo nuevo}. Son funciones para entrenar uno limpio en cada pliegue."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    return {
        "Regla (1 umbral)": lambda: DecisionTreeClassifier(max_depth=1, random_state=0),
        "Regresion logistica": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0),
    }


def cargar(csv_ruta, limite=90, bloque=10, incluir_copias=False):
    """
    Lee el CSV de extraer_puntos.py y prepara X (medidas) y Y (dos preguntas de si/no).
    - Las copias IDENTICAS (mismo hash) cuentan una sola vez: si no, la misma foto contaria doble y
      podria caer en entrenamiento y en prueba a la vez.
    - El grupo de cada foto = persona + categoria + bloque de 'bloque' fotos consecutivas (una sesion).
      Se recalcula aqui (no se usa la columna 'grupo' del CSV) para no juntar sesiones distintas por culpa
      de las fotos repetidas.
    """
    df = pd.read_csv(csv_ruta).sort_values("numero").reset_index(drop=True)
    df["copia"] = df.duplicated("hash", keep="first")
    base = np.where(df["numero"] <= limite, 0, limite)
    df["grupo"] = (df["persona"] + df["clase_original"].str[0]
                   + ((df["numero"] - 1 - base) // bloque).astype(str))
    n_copias = int(df["copia"].sum())
    if not incluir_copias:
        df = df[~df["copia"]]
    print(f"{n_copias} copias identicas " + ("se conservan" if incluir_copias else "descartadas (cada foto cuenta una vez)"))
    df = df[df["detectada"] == 1].reset_index(drop=True)
    X, Y = medidas_y_respuestas(df)
    return df, X, Y


def medidas_y_respuestas(df):
    """De un DataFrame con puntos p0_x.. y columna 'clase' saca X (medidas) y Y (las dos preguntas)."""
    X = np.array([caracteristicas([(r[f"p{i}_x"], r[f"p{i}_y"]) for i in range(33)])
                  for _, r in df.iterrows()])
    Y = {"abre_brazos": [], "abre_piernas": []}
    for c in df["clase"]:
        b, p = etiquetas_desde_clase(c)
        Y["abre_brazos"].append(b)
        Y["abre_piernas"].append(p)
    return X, {k: np.array(v) for k, v in Y.items()}


def cargar_capturas(rutas):
    """
    Lee uno o mas CSV de capturar_postura.py (tomas en vivo ya etiquetadas por el protocolo).
    'sesion' = nombre de la sesion (antes de _t<toma> en el nombre); 'grupo' = una toma.
    """
    df = pd.concat([pd.read_csv(r) for r in rutas], ignore_index=True)
    df = df[df["detectada"] == 1].reset_index(drop=True)
    df["sesion"] = df["archivo"].str.extract(r"^(.*)_t\d+_")[0]
    return df


def predecir_cv(fabrica, X, Y, grupos, pliegues=5):
    """Validacion cruzada por grupos: devuelve la prediccion de cada foto hecha por un modelo que no la vio."""
    from sklearn.model_selection import GroupKFold
    n_grupos = len(set(grupos))
    gkf = GroupKFold(n_splits=min(pliegues, n_grupos))
    pred = {k: np.zeros(len(X), dtype=int) for k in Y}
    for tr, te in gkf.split(X, groups=grupos):
        for k in Y:
            if len(set(Y[k][tr])) < 2:                 # el pliegue de entrenamiento no tiene las dos respuestas
                pred[k][te] = int(Y[k][tr][0])
                continue
            m = fabrica().fit(X[tr], Y[k][tr])
            pred[k][te] = m.predict(X[te])
    return pred


def predecir_entre(fabrica, X_ent, Y_ent, X_prueba):
    """Entrena con X_ent / Y_ent y predice X_prueba (una pregunta por cada llave de Y_ent)."""
    pred = {}
    for k in Y_ent:
        if len(set(Y_ent[k])) < 2:                     # en el entrenamiento todas las respuestas son iguales
            pred[k] = np.full(len(X_prueba), int(Y_ent[k][0]))
        else:
            pred[k] = fabrica().fit(X_ent, Y_ent[k]).predict(X_prueba)
    return pred


def metricas(esquema, modelo, Y, pred, idx):
    """Filas de resultados: accuracy y F1 de cada pregunta, y el acierto cuando las dos van bien a la vez."""
    from sklearn.metrics import accuracy_score, f1_score
    filas = []
    for k in OBJETIVOS:
        y, p = Y[k][idx], pred[k]
        filas.append({"esquema": esquema, "modelo": modelo, "pregunta": k, "n": len(idx),
                      "accuracy": accuracy_score(y, p), "f1_macro": f1_score(y, p, average="macro")})
    exacto = np.mean((Y["abre_brazos"][idx] == pred["abre_brazos"]) & (Y["abre_piernas"][idx] == pred["abre_piernas"]))
    filas.append({"esquema": esquema, "modelo": modelo, "pregunta": "las dos a la vez", "n": len(idx),
                  "accuracy": exacto, "f1_macro": np.nan})
    return filas


def clases_de(Y, pred, idx):
    """Convierte las dos preguntas si/no (reales y predichas) a la clase de cuatro (para la matriz de confusion)."""
    real = [clase_desde_etiquetas(b, p) for b, p in zip(Y["abre_brazos"][idx], Y["abre_piernas"][idx])]
    est = [clase_desde_etiquetas(b, p) for b, p in zip(pred["abre_brazos"], pred["abre_piernas"])]
    return real, est


def dibujar_matrices(resultados, esquema, ruta):
    """resultados: {modelo: (real, estimada)}. Una figura con las matrices de los tres modelos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    fig, ejes = plt.subplots(1, len(resultados), figsize=(4.6 * len(resultados), 4.4))
    ejes = np.atleast_1d(ejes)
    for ax, (modelo, (real, est)) in zip(ejes, resultados.items()):
        cm = confusion_matrix(real, est, labels=CLASES)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(modelo, fontsize=11)
        ax.set_xticks(range(len(CLASES)), CLASES, rotation=30, ha="right")
        ax.set_yticks(range(len(CLASES)), CLASES)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        for i in range(len(CLASES)):
            for j in range(len(CLASES)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.suptitle(esquema, fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    """Evalua los tres modelos (esquemas A, B, C y D), guarda tablas y matrices y, con --guardar, el modelo final."""
    ap = argparse.ArgumentParser(description="Entrena y evalua el clasificador de postura")
    ap.add_argument("--csv", default=os.path.join("data", "keypoints.csv"))
    ap.add_argument("--salida", default="results")
    ap.add_argument("--guardar", action="store_true", help="entrena con TODAS las fotos y guarda modelos\\postura.joblib")
    ap.add_argument("--modelo-final", default="Random forest", help="modelo que se guarda con --guardar")
    ap.add_argument("--limite", type=int, default=90, help="ultima foto de la categoria 'brazos' (por defecto 90)")
    ap.add_argument("--bloque", type=int, default=10, help="fotos consecutivas que forman una sesion (por defecto 10)")
    ap.add_argument("--incluir-copias", action="store_true", help="no descartar las fotos identicas repetidas")
    ap.add_argument("--captura", nargs="*", default=[], metavar="CSV",
                    help="CSV(s) de capturar_postura.py: se usan como prueba independiente (esquema C) y entrenamiento extra (D)")
    args = ap.parse_args()

    # --- cargar fotos (y capturas en vivo si las hay) ---
    df, X, Y = cargar(args.csv, args.limite, args.bloque, args.incluir_copias)
    cap = cargar_capturas(args.captura) if args.captura else None
    if cap is not None:
        Xc, Yc = medidas_y_respuestas(cap)
        print(f"capturas en vivo: {len(cap)} cuadros | clases: {cap['clase'].value_counts().to_dict()} | "
              f"sesiones: {cap['sesion'].nunique()} | tomas: {cap['grupo'].nunique()}")
    os.makedirs(args.salida, exist_ok=True)
    print(f"{len(df)} fotos con persona detectada | clases: {df['clase'].value_counts().to_dict()}")
    print(f"personas: {df['persona'].value_counts().to_dict()} | grupos: {df['grupo'].nunique()}")

    # --- evaluar los tres modelos en cada esquema ---
    modelos = fabricas_de_modelos()
    todas, matrices = [], {}
    todos = np.arange(len(df))

    # A. Validacion cruzada por grupos
    esq = "A. Validacion cruzada por grupos (5 pliegues)"
    matrices[esq] = {}
    for nombre, fab in modelos.items():
        pred = predecir_cv(fab, X, Y, df["grupo"].values)
        todas += metricas(esq, nombre, Y, pred, todos)
        matrices[esq][nombre] = clases_de(Y, pred, todos)

    # B. Entre personas (entrena con una y prueba con la otra)
    personas = sorted(df["persona"].unique())
    if len(personas) >= 2:
        for a, b in ((personas[0], personas[1]), (personas[1], personas[0])):
            tr = np.where(df["persona"] == a)[0]
            te = np.where(df["persona"] == b)[0]
            esq = f"B. Entrena con persona {a} ({len(tr)} fotos), prueba con persona {b} ({len(te)} fotos)"
            matrices[esq] = {}
            for nombre, fab in modelos.items():
                pred = predecir_entre(fab, X[tr], {k: v[tr] for k, v in Y.items()}, X[te])
                todas += metricas(esq, nombre, Y, pred, te)
                matrices[esq][nombre] = clases_de(Y, pred, te)
    else:
        print("Solo hay una persona en el CSV: se omite la evaluacion entre personas.")

    # C y D. Prueba con capturas en vivo (otra camara, otra persona, otro dia: la prueba mas honesta)
    if cap is not None:
        todos_c = np.arange(len(cap))
        esq = f"C. Entrena con las {len(df)} fotos, prueba con {len(cap)} cuadros en vivo"
        matrices[esq] = {}
        for nombre, fab in modelos.items():
            pred = predecir_entre(fab, X, Y, Xc)
            todas += metricas(esq, nombre, Yc, pred, todos_c)
            matrices[esq][nombre] = clases_de(Yc, pred, todos_c)

        # D. Fotos + capturas de entrenamiento; la prueba siempre son capturas que no se usaron para entrenar.
        # Con 2 o mas sesiones se deja una sesion fuera cada vez; con una sola, se deja una toma fuera.
        col = "sesion" if cap["sesion"].nunique() >= 2 else "grupo"
        unidad = "sesion" if col == "sesion" else "toma"
        esq = f"D. Fotos + capturas (se deja fuera una {unidad} en vivo cada vez)"
        matrices[esq] = {}
        for nombre, fab in modelos.items():
            pred_c = {k: np.zeros(len(cap), dtype=int) for k in OBJETIVOS}
            for fuera in cap[col].unique():
                te = np.where(cap[col] == fuera)[0]
                tr = np.where(cap[col] != fuera)[0]
                Xe = np.vstack([X, Xc[tr]])
                Ye = {k: np.concatenate([Y[k], Yc[k][tr]]) for k in OBJETIVOS}
                p = predecir_entre(fab, Xe, Ye, Xc[te])
                for k in OBJETIVOS:
                    pred_c[k][te] = p[k]
            todas += metricas(esq, nombre, Yc, pred_c, todos_c)
            matrices[esq][nombre] = clases_de(Yc, pred_c, todos_c)

    # --- guardar la tabla de metricas ---
    tabla = pd.DataFrame(todas)
    tabla.to_csv(os.path.join(args.salida, "clasificador_metricas.csv"), index=False)

    # Resumen en Markdown
    lineas = []
    for esq, sub in tabla.groupby("esquema", sort=False):
        lineas += [f"### {esq}", "", "| Modelo | Pregunta | n | Accuracy | F1 macro |", "|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            f1 = "-" if pd.isna(r["f1_macro"]) else f"{r['f1_macro']:.3f}"
            lineas.append(f"| {r['modelo']} | {r['pregunta']} | {int(r['n'])} | {r['accuracy']:.3f} | {f1} |")
        lineas.append("")
    with open(os.path.join(args.salida, "clasificador_resumen.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print("\n" + "\n".join(lineas))

    # Matrices de confusion: una imagen por esquema
    for i, (esq, res) in enumerate(matrices.items()):
        dibujar_matrices(res, esq, os.path.join(args.salida, f"matriz_confusion_{i + 1}.png"))

    # Importancia de cada medida (random forest entrenado con todo)
    filas = []
    for k in OBJETIVOS:
        rf = modelos["Random forest"]().fit(X, Y[k])
        for n, v in zip(NOMBRES, rf.feature_importances_):
            filas.append({"pregunta": k, "caracteristica": n, "importancia": v})
    pd.DataFrame(filas).sort_values(["pregunta", "importancia"], ascending=[True, False]).to_csv(
        os.path.join(args.salida, "importancia_caracteristicas.csv"), index=False)

    # Modelo final para usar en vivo (con --guardar)
    if args.guardar:
        import joblib
        os.makedirs("modelos", exist_ok=True)
        X_fin, Y_fin = X, Y
        if cap is not None:                      # el modelo final aprende de fotos + capturas
            X_fin = np.vstack([X, Xc])
            Y_fin = {k: np.concatenate([Y[k], Yc[k]]) for k in OBJETIVOS}
        final = {k: modelos[args.modelo_final]().fit(X_fin, Y_fin[k]) for k in OBJETIVOS}
        joblib.dump({"modelos": final, "caracteristicas": NOMBRES, "tipo": args.modelo_final},
                    os.path.join("modelos", "postura.joblib"))
        print("Modelo guardado en modelos\\postura.joblib")
    print(f"\nResultados en: {args.salida}\\")


if __name__ == "__main__":
    main()
