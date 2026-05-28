import numpy as np
from sklearn.tree import DecisionTreeClassifier
from export_tree import tree_to_dict, predict_dict


def _toy_model():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 5) * [100, 1, 18, 24, 53]
    y = (X[:, 0] + X[:, 4] > 80).astype(int)
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    return clf, X, y


def test_dict_tiene_estructura_esperada():
    clf, X, y = _toy_model()
    d = tree_to_dict(clf, ["edad", "sexo", "departamento", "anio", "semana"],
                     ["AGUDA", "ERUPTIVA"])
    assert d["feature_names"] == ["edad", "sexo", "departamento", "anio", "semana"]
    assert d["class_labels"] == ["AGUDA", "ERUPTIVA"]
    for key in ("children_left", "children_right", "feature", "threshold", "value"):
        assert key in d["nodes"]


def test_paridad_predict_dict_vs_sklearn():
    """El recorrido del dict exportado debe coincidir 100% con sklearn."""
    clf, X, y = _toy_model()
    d = tree_to_dict(clf, ["a", "b", "c", "d", "e"], ["0", "1"])
    sk = clf.predict(X)
    for i in range(len(X)):
        pred, prob = predict_dict(d, X[i])
        assert pred == sk[i], f"fila {i}: dict={pred} sklearn={sk[i]}"
