from pathlib import Path


def test_validation_configuration_exists():
    gx_config = Path("gx/great_expectations.yml")

    assert gx_config.exists(), "Great Expectations configuration is missing"
