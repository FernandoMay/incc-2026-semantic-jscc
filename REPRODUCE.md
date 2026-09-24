# SD-02 Reproduction Protocol

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
python src/simulator.py
```

The current implementation is a PCA-based semantic JSCC prototype over a synthetic Rayleigh channel. It is not yet a trained Deep JSCC model and must not be reported as one. The final paper requires public image/task data, matched raw/codec baselines, repeated seeds, confidence intervals, and a calibrated physical/channel model.
