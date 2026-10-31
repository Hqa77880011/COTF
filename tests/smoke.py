"""CPU software checks on analytic fields; no paper-performance assertions."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotf import cli, method

torch.set_num_threads(1)
out = Path("runs/smoke")
out.mkdir(parents=True, exist_ok=True)
def command(*args):
    sys.argv = ["cotf", *map(str, args)]
    cli.main()
command("demo", "--output", out/"source.npz", "--count", 128)
command("prepare", "--input", out/"source.npz", "--output", out/"data")
command("train", "--input", out/"data", "--output", out/"fit", "--config", "config.json", "--epochs", 2)
command("calibrate", "--input", out/"data/calibration.npz", "--model", out/"fit/model.pt", "--output", out/"fit/calibrated.pt")
command("evaluate", "--input", out/"data/test.npz", "--model", out/"fit/calibrated.pt", "--output", out/"fit/evaluation")
command("plot", "--input", out/"fit", "--output", out/"figures")
state = torch.load(out/"fit/model.pt", weights_only=True)
data = cli.read(out/"data/train.npz")
batch = cli.tensor_batch(data, np.arange(4), "cpu")
model = method.Model(state["config"])
model.load_state_dict(state["state"])
model.eval()
before = model(batch)["prediction"].detach()
clone = method.Model(state["config"])
clone.load_state_dict(state["state"])
clone.eval()
assert torch.allclose(before, clone(batch)["prediction"], atol=1e-6)
optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
loss = model.objective(batch, model(batch))
loss.backward()
assert torch.isfinite(loss)
assert model.backbone.token.weight.grad.abs().sum() > 0
assert model.backbone.output[-1].weight.grad.abs().sum() > 0
old = model.backbone.token.weight.detach().clone()
optimizer.step()
assert not torch.equal(old, model.backbone.token.weight)
# Hand calculation: sorted errors 1,3 -> risks 1,2 -> AURC 1.5.
p = {"prediction": np.array([[1., 3.]]), "trust_risk": np.array([[.1, .9]]),
     "gain": np.ones((1,2)), "fragility": np.zeros((1,2)),
     "lower": np.array([[0., 0.]]), "upper": np.array([[2., 2.]])}
d = {"y": np.zeros((1,2)), "query": np.array([[[0.,0.],[1.,0.]]]), "group": np.array([0])}
m = method.measure(p,d,{})
assert np.isclose(m["aurc"], 1.5) and m["whole_object_coverage"] == 1
print(json.dumps({"loss":float(loss.detach()), "checkpoint_max_error":float((before-clone(batch)["prediction"]).abs().max()), "metrics":json.loads((out/"fit/evaluation/metrics.json").read_text())}))
