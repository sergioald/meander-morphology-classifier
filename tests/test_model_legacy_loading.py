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
