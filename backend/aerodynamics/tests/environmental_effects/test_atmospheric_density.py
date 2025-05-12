import pytest
from environmental_effects.atmospheric_density import (
    density_at_altitude as atmospheric_density,
)

# Tolerance for relative comparisons (1% error margin)
REL_TOL = 1e-2
# Altitudes for testing (in meters)
SEA_LEVEL = 0
TEN_KM = 10_000
TWENTY_KM = 20_000
HIGH_ALTITUDE = 100_000  # 100 km, above typical atmospheric model range


@pytest.mark.parametrize(
    "altitude_m, expected_density",
    [
        # Standard reference: sea level density ~1.225 kg/m^3
        (SEA_LEVEL, 1.225),
        # Exponential model outputs from density_at_altitude for reference
        (TEN_KM, 0.3777473306733123),
        (TWENTY_KM, 0.1164841190455614),
    ],
    ids=["sea_level", "10km_exponential", "20km_exponential"],
)
def test_known_altitude_values(altitude_m, expected_density):
    """
    Verify that the density at known altitudes matches expected values.
    Uses a relative tolerance to account for model rounding.
    """
    density = atmospheric_density(altitude_m)
    # Confirm result is within 1% of expected
    assert (
        pytest.approx(expected_density, rel=REL_TOL) == density
    ), f"Density at {altitude_m} m: expected ~{expected_density}, got {density}"


def test_negative_altitude_clamped_to_sea_level():
    """
    The function should treat any negative altitude as sea level (0 m).
    """
    density_negative = atmospheric_density(-500)
    density_sea = atmospheric_density(SEA_LEVEL)
    # Exact match within tiny tolerance
    assert density_negative == pytest.approx(
        density_sea, rel=1e-6
    ), "Negative altitudes must clamp to sea level density"


def test_high_altitude_behavior():
    """
    At extremely high altitudes, the density should not be negative
    and should not exceed the density measured at lower altitudes.
    """
    density_high = atmospheric_density(HIGH_ALTITUDE)
    density_20km = atmospheric_density(TWENTY_KM)

    # Density must remain non-negative
    assert density_high >= 0, "Density should never be negative"
    # Density should decrease with altitude
    assert (
        density_high <= density_20km
    ), f"Density at {HIGH_ALTITUDE} m ({density_high}) should be <= density at 20 km ({density_20km})"


def test_invalid_type_raises_type_error():
    """
    Passing a non-numeric input should raise a TypeError to
    enforce correct usage of the function.
    """
    with pytest.raises(TypeError):
        atmospheric_density(None)
    with pytest.raises(TypeError):
        atmospheric_density("1000")
