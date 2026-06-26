import numpy as np
import pandas as pd

from meander_morphology.compound_model import (
    encode_spectra_with_encoder,
    prepare_spectra_batch,
    save_latent_outputs,
)


class DummyEncoder:
    def predict(self, batch, batch_size=128, verbose=0):
        del batch_size, verbose
        flat = batch.reshape((batch.shape[0], -1))
        return np.column_stack([flat.mean(axis=1), flat.std(axis=1)])


def test_prepare_spectra_batch_adds_channel_dimension():
    spectra = np.zeros((3, 32, 32), dtype=float)
    batch = prepare_spectra_batch(spectra)
    assert batch.shape == (3, 32, 32, 1)
    assert batch.dtype == np.float32


def test_encode_spectra_with_dummy_encoder_returns_latent_coordinates():
    spectra = np.zeros((2, 8, 8), dtype=float)
    spectra[1] = 1.0
    latent = encode_spectra_with_encoder(DummyEncoder(), spectra)
    assert latent.shape == (2, 2)
    assert np.allclose(latent[:, 0], [0.0, 1.0])


def test_save_latent_outputs_merges_summary(tmp_path):
    summary = pd.DataFrame(
        {
            "unit_id": [0, 1],
            "n_lobes": [1, 3],
            "is_compound": [False, True],
        }
    )
    summary_path = tmp_path / "compound_bend_summary.csv"
    summary.to_csv(summary_path, index=False)

    latent = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = save_latent_outputs(latent, tmp_path / "encoded", summary_path=summary_path)

    assert result.latent_path.exists()
    assert result.table_path.exists()
    saved = pd.read_csv(result.table_path)
    assert list(saved.columns) == ["unit_id", "n_lobes", "is_compound", "latent_1", "latent_2"]
    assert np.allclose(np.load(result.latent_path), latent)
