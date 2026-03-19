"""
Named literature-backed forcing presets for Europa2D.

These presets provide reproducible labels and default amplitudes for the
latitude-only ocean heat-flux proxies used by the 2D shell model.

References:
    Lemasquerier et al. (2023): q* contrast parameter, DOI: 10.1029/2023AV000994
    Soderlund et al. (2014): low-latitude ocean heating, DOI: 10.1038/ngeo2021
    Ashkenazy & Tziperman (2021): efficient meridional transport, DOI: 10.1038/s41467-021-26710-0
"""
from dataclasses import dataclass
from typing import Literal

from latitude_profile import LatitudeProfile, OceanPattern


ScenarioName = Literal[
    "uniform_transport",
    "soderlund2014_equator",
    "lemasquerier2023_polar",
    "lemasquerier2023_polar_strong",
]


@dataclass(frozen=True)
class LiteratureScenario:
    """Container for a named literature-backed forcing preset."""

    name: ScenarioName
    title: str
    citation: str
    reference_url: str
    description: str
    ocean_pattern: OceanPattern
    q_star: float

    def build_profile(
        self,
        T_eq: float,
        epsilon_eq: float,
        epsilon_pole: float,
        q_ocean_mean: float,
        T_floor: float = 52.0,
    ) -> LatitudeProfile:
        """Create a LatitudeProfile using the preset forcing family."""
        return LatitudeProfile(
            T_eq=T_eq,
            epsilon_eq=epsilon_eq,
            epsilon_pole=epsilon_pole,
            q_ocean_mean=q_ocean_mean,
            ocean_pattern=self.ocean_pattern,
            q_star=self.q_star if self.q_star > 0 else None,
            T_floor=T_floor,
            strict_q_star=False,  # scenarios are pre-validated
        )


SCENARIOS: dict[ScenarioName, LiteratureScenario] = {
    "uniform_transport": LiteratureScenario(
        name="uniform_transport",
        title="Uniform transport proxy",
        citation="Ashkenazy & Tziperman (2021)",
        reference_url="https://www.nature.com/articles/s41467-021-26710-0",
        description="Efficient meridional transport benchmark with no imposed latitude contrast.",
        ocean_pattern="uniform",
        q_star=0.0,
    ),
    "soderlund2014_equator": LiteratureScenario(
        name="soderlund2014_equator",
        title="Equator-enhanced proxy",
        citation="Soderlund et al. (2014)",
        reference_url="https://doi.org/10.1038/ngeo2021",
        description="Low-latitude ocean heat-delivery benchmark with q* = 0.4.",
        ocean_pattern="equator_enhanced",
        q_star=0.4,
    ),
    "lemasquerier2023_polar": LiteratureScenario(
        name="lemasquerier2023_polar",
        title="Polar-enhanced proxy",
        citation="Lemasquerier et al. (2023)",
        reference_url="https://doi.org/10.1029/2023AV000994",
        description="Conservative polar-enhanced mantle-tidal benchmark, mantle_tidal_fraction = 0.5.",
        ocean_pattern="polar_enhanced",
        q_star=0.455,
    ),
    "lemasquerier2023_polar_strong": LiteratureScenario(
        name="lemasquerier2023_polar_strong",
        title="Strong polar-enhanced proxy",
        citation="Lemasquerier et al. (2023)",
        reference_url="https://doi.org/10.1029/2023AV000994",
        description="Upper-end polar-tidal sensitivity, mantle_tidal_fraction = 0.9.",
        ocean_pattern="polar_enhanced",
        q_star=0.819,
    ),
}

DEFAULT_SCENARIO: ScenarioName = "lemasquerier2023_polar"


def get_scenario(name: ScenarioName) -> LiteratureScenario:
    """Return a named literature-backed forcing preset."""
    return SCENARIOS[name]


def list_scenarios() -> tuple[ScenarioName, ...]:
    """Return the available preset names in stable order."""
    return tuple(SCENARIOS.keys())
