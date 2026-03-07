import os

import numpy as np

from app.triton_client import MODEL_PROFILES, TritonClient


def _client() -> TritonClient:
    original = os.environ.get("TRITON_MOCK")
    os.environ["TRITON_MOCK"] = "true"
    try:
        return TritonClient()
    finally:
        if original is None:
            os.environ.pop("TRITON_MOCK", None)
        else:
            os.environ["TRITON_MOCK"] = original


def test_prepare_triton_input_normalizes_anime_model_to_neg_one_to_one() -> None:
    client = _client()
    profile = MODEL_PROFILES["anime_gan_hayao"]
    chw = np.array(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.9, 0.8], [0.7, 0.6]],
        ],
        dtype=np.float32,
    )

    prepared = client._prepare_triton_input(chw, profile)

    assert prepared.shape == (1, 2, 2, 3)
    np.testing.assert_allclose(prepared[0, 0, 0], [-1.0, -0.8, 0.8])
    np.testing.assert_allclose(prepared[0, 0, 1], [0.0, -0.6, 0.6])
    np.testing.assert_allclose(prepared[0, 1, 0], [1.0, -0.4, 0.4])


def test_normalize_output_maps_anime_tanh_tensor_back_to_zero_to_one() -> None:
    client = _client()
    profile = MODEL_PROFILES["anime_gan_hayao"]
    output = np.array(
        [
            [
                [[-1.0, -0.5, 0.0], [0.5, 1.0, -1.0]],
                [[0.25, -0.25, 0.75], [-0.75, 0.0, 1.0]],
            ]
        ],
        dtype=np.float32,
    )

    normalized = client._normalize_output(output, profile)

    assert normalized.shape == (3, 2, 2)
    np.testing.assert_allclose(normalized[:, 0, 0], [0.0, 0.25, 0.5])
    np.testing.assert_allclose(normalized[:, 0, 1], [0.75, 1.0, 0.0])
    np.testing.assert_allclose(normalized[:, 1, 0], [0.625, 0.375, 0.875])


def test_prepare_onnx_output_to_chw_uses_model_specific_output_range() -> None:
    client = _client()
    client.local_output_shapes["anime_gan_hayao"] = [1, 2, 2, 3]
    output = np.array(
        [
            [
                [[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.0]],
                [[0.25, -0.25, 0.75], [1.0, -1.0, 0.5]],
            ]
        ],
        dtype=np.float32,
    )

    normalized = client._prepare_onnx_output_to_chw("anime_gan_hayao", output)

    assert normalized.shape == (3, 2, 2)
    np.testing.assert_allclose(normalized[:, 0, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(normalized[:, 1, 1], [1.0, 0.0, 0.75])
