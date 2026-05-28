"""Exporta un DecisionTreeClassifier de sklearn a un dict/JSON serializable,
y provee un recorrido `predict_dict` idéntico al que usará el navegador.

El recorrido en JS (public/index.html) es una traducción literal de
`predict_dict`, así que el test de paridad garantiza que la web == Python.
"""
import json


def tree_to_dict(clf, feature_names, class_labels):
    t = clf.tree_
    return {
        "feature_names": list(feature_names),
        "class_labels": list(class_labels),
        "nodes": {
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            # value: (n_nodes, 1, n_classes) -> conteos por clase en cada nodo
            "value": t.value.tolist(),
        },
    }


def predict_dict(tree_dict, x):
    """Recorre el árbol exportado. Devuelve (pred_clase, prob_clase1).

    Réplica EXACTA de la lógica JS. Una hoja en sklearn tiene
    children_left[node] == -1. El desempate replica np.argmax de sklearn:
    en empate de conteos se elige la clase 0.
    """
    n = tree_dict["nodes"]
    cl, cr = n["children_left"], n["children_right"]
    feat, thr, val = n["feature"], n["threshold"], n["value"]
    node = 0
    while cl[node] != -1:
        if x[feat[node]] <= thr[node]:
            node = cl[node]
        else:
            node = cr[node]
    counts = val[node][0]
    total = counts[0] + counts[1]
    prob1 = counts[1] / total if total else 0.0
    pred = 1 if counts[1] > counts[0] else 0  # argmax (desempata a clase 0)
    return pred, prob1


def save_json(tree_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f)
