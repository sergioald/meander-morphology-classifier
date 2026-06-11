import pytest

from meander_morphology.model import _remove_default_groups_from_config, prepare_images_for_keras


def test_remove_default_groups_from_nested_config():
    config = {
        "class_name": "Functional",
        "config": {
            "layers": [
                {
                    "class_name": "Conv2DTranspose",
                    "config": {"filters": 16, "groups": 1},
                },
                {
                    "class_name": "Dense",
                    "config": {"units": 2},
                },
            ]
        },
    }
    cleaned = _remove_default_groups_from_config(config)
    assert "groups" not in cleaned["config"]["layers"][0]["config"]
    assert cleaned["config"]["layers"][0]["config"]["filters"] == 16


def test_prepare_images_for_keras_adds_channel_axis():
    x = prepare_images_for_keras([[1, 2], [3, 4]])
    assert x.shape == (1, 2, 2, 1)


@pytest.mark.optional
def test_build_encoder_from_nested_autoencoder_when_tensorflow_available():
    tf = pytest.importorskip("tensorflow")
    from meander_morphology.model import build_encoder_from_autoencoder

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(64, 64, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, name="latent"),
        ],
        name="encoder",
    )
    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(2,)),
            tf.keras.layers.Dense(64 * 64),
            tf.keras.layers.Reshape((64, 64, 1)),
        ],
        name="decoder",
    )
    autoencoder = tf.keras.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))
    recovered = build_encoder_from_autoencoder(autoencoder, latent_dim=2)
    assert recovered.output_shape[-1] == 2
