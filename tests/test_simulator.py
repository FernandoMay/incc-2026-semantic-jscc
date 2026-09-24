from src.simulator import evaluate


def test_evaluation_schema():
    result = evaluate(seed=7, snr_values=(0, 5), latent_dim=8)
    assert result["samples"] > 0
    assert set(result["snr"]) == {"0", "5"}


def test_reproducible():
    assert evaluate(seed=9, snr_values=(2,), latent_dim=8) == evaluate(seed=9, snr_values=(2,), latent_dim=8)
