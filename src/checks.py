def validate_config(config):
    """
    validates if config is valid. keep adding validation as config grows complex.
    """

    assert "I" in config, "Missing key: I"
    assert "features" in config, "Missing key: features"
    assert "outcome_model" in config, "Missing key: outcome_model"

    features = config["features"]
    outcome_model = config["outcome_model"]
    objective = config.get("objective", [])

    M = len(features)

    # Validate feature distributions
    for name, spec in features.items():

        dist = spec["dist"]

        if dist == "normal":
            assert (
                "mean" in spec and "std" in spec
            ), f"Feature '{name}' missing mean/std"

        elif dist == "uniform":
            assert (
                "low" in spec and "high" in spec
            ), f"Feature '{name}' missing low/high"

        elif dist == "bernoulli":
            assert "p" in spec, f"Feature '{name}' missing p"

        else:
            raise ValueError(f"Unknown distribution: {dist}")

    baseline_coeff = outcome_model.get("baseline_coeff")
    treatment_coeff = outcome_model.get("treatment_coeff")

    assert baseline_coeff is not None, "baseline_coeff missing"
    assert treatment_coeff is not None, "treatment_coeff missing"

    assert (
        len(baseline_coeff) == M + 1
    ), "baseline_coeff length must be M+1 (including intercept)"

    assert (
        len(treatment_coeff) == M + 1
    ), "treatment_coeff length must be M+1 (including intercept)"

    if "profit" in objective:
        assert (
            len(config.get("profit_coeff", [])) == M + 1
        ), "profit_coeff must have length M+1"

    if "cost" in objective:
        assert (
            len(config.get("cost_coeff", [])) == M + 1
        ), "cost_coeff must have length M+1"

    return "config is valid"

    assert "I" in config
    assert "features" in config
    assert "outcome_model" in config

    features = config["features"]
    outcome_model = config["outcome_model"]
    objective = config.get("objective", [])

    M = len(features)

    for name, spec in features.items():

        dist = spec["dist"]

        if dist == "normal":
            assert "mean" in spec and "std" in spec

        elif dist == "uniform":
            assert "low" in spec and "high" in spec

        elif dist == "bernoulli":
            assert "p" in spec

        else:
            raise ValueError(f"Unknown distribution: {dist}")

    baseline_coeff = outcome_model.get("baseline_coeff")
    treatment_coeff = outcome_model.get("treatment_coeff")

    assert len(baseline_coeff) == M + 1
    assert len(treatment_coeff) == M + 1

    if "profit" in objective:
        assert len(config.get("profit_coeff", [])) == M + 1

    if "cost" in objective:
        assert len(config.get("cost_coeff", [])) == M + 1

    return "config is valid"
