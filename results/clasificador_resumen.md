### A. Validacion cruzada por grupos (5 pliegues)

| Modelo | Pregunta | n | Accuracy | F1 macro |
|---|---|---|---|---|
| Regla (1 umbral) | abre_brazos | 159 | 0.679 | 0.647 |
| Regla (1 umbral) | abre_piernas | 159 | 0.767 | 0.755 |
| Regla (1 umbral) | las dos a la vez | 159 | 0.516 | - |
| Regresion logistica | abre_brazos | 159 | 0.755 | 0.751 |
| Regresion logistica | abre_piernas | 159 | 0.717 | 0.716 |
| Regresion logistica | las dos a la vez | 159 | 0.648 | - |
| Random forest | abre_brazos | 159 | 0.692 | 0.689 |
| Random forest | abre_piernas | 159 | 0.767 | 0.764 |
| Random forest | las dos a la vez | 159 | 0.642 | - |

### B. Entrena con persona A (104 fotos), prueba con persona B (55 fotos)

| Modelo | Pregunta | n | Accuracy | F1 macro |
|---|---|---|---|---|
| Regla (1 umbral) | abre_brazos | 55 | 0.509 | 0.503 |
| Regla (1 umbral) | abre_piernas | 55 | 0.491 | 0.437 |
| Regla (1 umbral) | las dos a la vez | 55 | 0.327 | - |
| Regresion logistica | abre_brazos | 55 | 0.836 | 0.821 |
| Regresion logistica | abre_piernas | 55 | 0.855 | 0.848 |
| Regresion logistica | las dos a la vez | 55 | 0.800 | - |
| Random forest | abre_brazos | 55 | 0.782 | 0.752 |
| Random forest | abre_piernas | 55 | 0.727 | 0.724 |
| Random forest | las dos a la vez | 55 | 0.618 | - |

### B. Entrena con persona B (55 fotos), prueba con persona A (104 fotos)

| Modelo | Pregunta | n | Accuracy | F1 macro |
|---|---|---|---|---|
| Regla (1 umbral) | abre_brazos | 104 | 0.702 | 0.684 |
| Regla (1 umbral) | abre_piernas | 104 | 0.615 | 0.615 |
| Regla (1 umbral) | las dos a la vez | 104 | 0.558 | - |
| Regresion logistica | abre_brazos | 104 | 0.596 | 0.591 |
| Regresion logistica | abre_piernas | 104 | 0.644 | 0.604 |
| Regresion logistica | las dos a la vez | 104 | 0.519 | - |
| Random forest | abre_brazos | 104 | 0.702 | 0.702 |
| Random forest | abre_piernas | 104 | 0.769 | 0.760 |
| Random forest | las dos a la vez | 104 | 0.635 | - |

### C. Entrena con las 159 fotos, prueba con 3900 cuadros en vivo

| Modelo | Pregunta | n | Accuracy | F1 macro |
|---|---|---|---|---|
| Regla (1 umbral) | abre_brazos | 3900 | 0.714 | 0.696 |
| Regla (1 umbral) | abre_piernas | 3900 | 0.443 | 0.342 |
| Regla (1 umbral) | las dos a la vez | 3900 | 0.263 | - |
| Regresion logistica | abre_brazos | 3900 | 0.622 | 0.617 |
| Regresion logistica | abre_piernas | 3900 | 0.567 | 0.559 |
| Regresion logistica | las dos a la vez | 3900 | 0.342 | - |
| Random forest | abre_brazos | 3900 | 0.651 | 0.650 |
| Random forest | abre_piernas | 3900 | 0.746 | 0.746 |
| Random forest | las dos a la vez | 3900 | 0.479 | - |

### D. Fotos + capturas (se deja fuera una sesion en vivo cada vez)

| Modelo | Pregunta | n | Accuracy | F1 macro |
|---|---|---|---|---|
| Regla (1 umbral) | abre_brazos | 3900 | 0.644 | 0.636 |
| Regla (1 umbral) | abre_piernas | 3900 | 0.691 | 0.681 |
| Regla (1 umbral) | las dos a la vez | 3900 | 0.487 | - |
| Regresion logistica | abre_brazos | 3900 | 0.752 | 0.744 |
| Regresion logistica | abre_piernas | 3900 | 0.714 | 0.712 |
| Regresion logistica | las dos a la vez | 3900 | 0.544 | - |
| Random forest | abre_brazos | 3900 | 0.690 | 0.671 |
| Random forest | abre_piernas | 3900 | 0.738 | 0.738 |
| Random forest | las dos a la vez | 3900 | 0.521 | - |
