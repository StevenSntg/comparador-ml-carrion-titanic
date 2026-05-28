// Verifica que la lógica JS del árbol coincide con la estructura del JSON real.
import fs from "node:fs";
import assert from "node:assert";

// --- réplica EXACTA de predictTree que irá en index.html ---
function predictTree(tree, x) {
  const n = tree.nodes;
  let node = 0;
  while (n.children_left[node] !== -1) {
    node = x[n.feature[node]] <= n.threshold[node]
      ? n.children_left[node] : n.children_right[node];
  }
  const counts = n.value[node][0];
  const total = counts[0] + counts[1];
  const prob1 = total ? counts[1] / total : 0;
  return { pred: counts[1] > counts[0] ? 1 : 0, prob: prob1 };  // argmax
}

const tree = JSON.parse(fs.readFileSync("public/models/titanic/ad.json", "utf8"));
assert.strictEqual(tree.feature_names.length, 5);
// raíz no es hoja
assert.ok(tree.nodes.children_left[0] !== -1);
// un caso cualquiera devuelve 0 o 1
const r = predictTree(tree, [83.475, 1, 1, 35, 1]);
assert.ok(r.pred === 0 || r.pred === 1);
console.log("OK predictTree:", r);
