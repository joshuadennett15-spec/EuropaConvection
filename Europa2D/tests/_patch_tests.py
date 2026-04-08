"""Patch test_monte_carlo_2d.py to add missing T_c/Ti statistics."""
import pathlib

p = pathlib.Path(__file__).parent / "test_monte_carlo_2d.py"
c = p.read_text(encoding="utf-8")
changed = False

old1 = """        H_mean=np.zeros(5),\n        H_sigma_low=np.zeros(5),\n        H_sigma_high=np.zeros(5),\n        runtime_seconds=0.0,\n        ocean_pattern="uniform",\n        ocean_amplitude=0.0,\n    )"""
new1 = """        H_mean=np.zeros(5),\n        H_sigma_low=np.zeros(5),\n        H_sigma_high=np.zeros(5),\n        runtime_seconds=0.0,\n        ocean_pattern="uniform",\n        ocean_amplitude=0.0,\n        T_c_median=np.zeros(5),\n        T_c_mean=np.zeros(5),\n        Ti_median=np.zeros(5),\n        Ti_mean=np.zeros(5),\n    )"""

if old1 in c:
    c = c.replace(old1, new1, 1)
    changed = True
    print("  [1] Added T_c/Ti to mock initialization")

if changed:
    p.write_text(c, encoding="utf-8")
    print("All patches applied.")
else:
    print("No patches applied.")
