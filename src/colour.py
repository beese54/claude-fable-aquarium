"""The project's display transform, shared by plate prep and the final comp.

Renders are scene-linear. The filmed plate is display-referred sRGB. To put both
through one pipeline, the plate is taken back to scene-linear with the inverse of
the same curve that will later be applied to the render, so plate pixels come out
of the comp exactly as they were filmed while rendered highlights roll off the
way the plate's do.

display = srgb_oetf(shoulder(linear))
shoulder(x) = x                              for x <= KNEE
            = KNEE + S * (1 - exp(-(x-KNEE)/S))  with S = 1 - KNEE, for x > KNEE
"""
import numpy as np

KNEE = 0.6
_S = 1.0 - KNEE
_CLIP = 0.9995   # display white maps to a finite linear value (about 3.6)


def srgb_oetf(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_eotf(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def shoulder(x):
    x = np.maximum(x, 0.0)
    return np.where(x <= KNEE, x, KNEE + _S * (1.0 - np.exp(-(x - KNEE) / _S)))


def shoulder_inv(y):
    y = np.clip(y, 0.0, _CLIP)
    return np.where(y <= KNEE, y, KNEE - _S * np.log(1.0 - (y - KNEE) / _S))


def linear_to_display(x):
    """Scene-linear render values -> display sRGB in 0..1."""
    return srgb_oetf(shoulder(x))


def display_to_linear(v):
    """Display sRGB in 0..1 (the plate) -> scene-linear, exact inverse of linear_to_display."""
    return shoulder_inv(srgb_eotf(v))


if __name__ == "__main__":
    v = np.linspace(0, 1, 4097)
    err = np.abs(linear_to_display(display_to_linear(v)) - v)
    print(f"round-trip max error below clip: {err[v < 0.999].max():.2e}; display white -> linear {display_to_linear(1.0):.2f}")
    assert err[v < 0.999].max() < 1e-6
