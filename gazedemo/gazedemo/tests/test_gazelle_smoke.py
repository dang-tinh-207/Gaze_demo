"""Real Gazelle on the GPU (skipped by -m "not gpu"): load, one image → heatmap (64, 64)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from gazedemo.config import load_config
    from gazedemo.perception.gazelle_model import GazelleModel
    return GazelleModel.from_config(load_config().gazelle, warmup=1)


@pytest.fixture(scope="module")
def image():
    from gazedemo.config import load_config, project_path
    from gazedemo.io.logs import load_image
    repo = load_config().gazelle.repo
    path = project_path(repo) / "assets" / "succession.png" if repo else None
    img = load_image(path) if path and path.exists() else None
    if img is None:
        img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    return img[:, :, ::-1].copy()  # → RGB


def test_heatmap_shape_and_range(model, image):
    hm, inout = model.infer(image)
    assert hm.shape == (64, 64) and hm.dtype == np.float32
    assert 0.0 <= hm.min() and hm.max() <= 1.0
    assert (inout is None) == (not model.has_inout)
    if inout is not None:
        assert 0.0 <= inout <= 1.0


def test_gpu_preprocess_matches_reference_pil_path(model, image):
    x_fast = model.to_input(image).float().cpu()
    steps, model._steps = model._steps, None
    try:
        x_ref = model.to_input(image).float().cpu()
        hm_ref, _ = model.infer(image)
    finally:
        model._steps = steps
    hm_fast, _ = model.infer(image)
    assert x_fast.shape == x_ref.shape == (1, 3, 448, 448)
    assert float((x_fast - x_ref).abs().max()) < 1e-3
    assert np.argmax(hm_fast) == np.argmax(hm_ref)
