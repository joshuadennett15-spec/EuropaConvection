"""Patch Convection.py to add use_composite_transition_closure flag."""
import pathlib

p = pathlib.Path(__file__).parent / "EuropaProjectDJ/src/Convection.py"
c = p.read_text(encoding="utf-8")
changed = False

# --- 1. green_cond_base_temp signature ---
old1 = (
    "            T_melt: float,\n"
    "            T_surface: float,\n"
    "            Q_v: float,\n"
    "            eta_ref: float = Rheology.VISCOSITY_REF,\n"
    "            N: float = 1.0,\n"
    "    ) -> Tuple[float, float]:"
)
new1 = (
    "            T_melt: float,\n"
    "            T_surface: float,\n"
    "            Q_v: float,\n"
    "            eta_ref: float = Rheology.VISCOSITY_REF,\n"
    "            N: float = 1.0,\n"
    "            use_composite_transition_closure: bool = False,\n"
    "            d_grain: Optional[float] = None,\n"
    "            d_del: Optional[float] = None,\n"
    "            D0v: Optional[float] = None,\n"
    "            D0b: Optional[float] = None,\n"
    "            Q_b: Optional[float] = None,\n"
    "            p_grain: Optional[float] = None,\n"
    "    ) -> Tuple[float, float]:"
)
if old1 in c:
    c = c.replace(old1, new1, 1)
    changed = True
    print("  [1] Patched green_cond_base_temp signature")

# --- 2. green_cond_base_temp logic ---
old2 = (
    "        # Step 2: Viscous temperature scale from rheology\n"
    "        # A = E / (N * R * Tm)\n"
    "        # ni = nm * exp(A * ((Tm/Ti) - 1))\n"
    "        # dni/dTi = -nm * (A*Tm/Ti^2) * exp(A * ((Tm/Ti) - 1))\n"
    "        # DTv = -ni / (dni/dTi)\n"
    "        A = Q_v / (N * R * T_melt)\n"
    "        \n"
    "        # Safeguard: limit exponent to prevent overflow (exp(700) ≈ 10^304)\n"
    "        exponent = A * ((T_melt / Ti) - 1)\n"
    "        exponent = np.clip(exponent, -500, 500)\n"
    "        \n"
    "        exp_term = np.exp(exponent)\n"
    "        ni = eta_ref * exp_term\n"
    "        dni_dTi = -eta_ref * (A * T_melt / Ti**2) * exp_term\n"
    "        \n"
    "        # Safeguard: prevent division by zero/inf\n"
    "        if np.abs(dni_dTi) < 1e-100 or not np.isfinite(dni_dTi):\n"
    "            # Fallback to Howell method\n"
    "            Tc = IceConvection.howell_cond_base_temp(T_melt, Q_v)\n"
    "            Tc = np.clip(Tc, T_surface + 1.0, T_melt - 1.0)\n"
    "            return Tc, Ti\n"
    "        \n"
    "        DTv = -ni / dni_dTi"
)

new2 = (
    "        # Step 2: Viscous temperature scale from rheology\n"
    "        fallback_to_analytic = True\n"
    "        DTv = 0.0\n"
    "        \n"
    "        if use_composite_transition_closure:\n"
    "            # Use central difference on log(eta) to find viscous temperature scale\n"
    "            # DTv = -1 / (d ln(eta) / dT)\n"
    "            dT = 0.1  # K\n"
    "            eta_plus = IcePhysics.composite_viscosity(\n"
    "                Ti + dT, d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,\n"
    "                Q_diff=Q_v, Q_gbs=Q_b, p_grain=p_grain\n"
    "            )\n"
    "            eta_minus = IcePhysics.composite_viscosity(\n"
    "                Ti - dT, d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,\n"
    "                Q_diff=Q_v, Q_gbs=Q_b, p_grain=p_grain\n"
    "            )\n"
    "            \n"
    "            # Safe log difference. If viscosity is clipped (e.g. at 1e12 or 1e25),\n"
    "            # the derivative will be exactly 0.0, and we must fall back.\n"
    "            if eta_plus > 0 and eta_minus > 0:\n"
    "                dlneta_dTi = (np.log(eta_plus) - np.log(eta_minus)) / (2 * dT)\n"
    "                if np.isfinite(dlneta_dTi) and abs(dlneta_dTi) > 1e-10:\n"
    "                    DTv = -1.0 / dlneta_dTi\n"
    "                    fallback_to_analytic = False\n"
    "        \n"
    "        if fallback_to_analytic:\n"
    "            A = Q_v / (N * R * T_melt)\n"
    "            exponent = np.clip(A * ((T_melt / Ti) - 1), -500, 500)\n"
    "            exp_term = np.exp(exponent)\n"
    "            ni = eta_ref * exp_term\n"
    "            dni_dTi = -eta_ref * (A * T_melt / Ti**2) * exp_term\n"
    "            \n"
    "            if np.abs(dni_dTi) < 1e-100 or not np.isfinite(dni_dTi):\n"
    "                Tc = IceConvection.howell_cond_base_temp(T_melt, Q_v)\n"
    "                Tc = np.clip(Tc, T_surface + 1.0, T_melt - 1.0)\n"
    "                return Tc, Ti\n"
    "            \n"
    "            DTv = -ni / dni_dTi"
)
if old2 in c:
    c = c.replace(old2, new2, 1)
    changed = True
    print("  [2] Patched green_cond_base_temp logic")

# --- 3. compute_transition_temperature signature ---
old3 = (
    "            eta_ref: float = Rheology.VISCOSITY_REF,\n"
    "            default_T_c: float = 250.0,\n"
    "            use_green_method: bool = True,\n"
    "    ) -> Tuple[float, float]:"
)
new3 = (
    "            eta_ref: float = Rheology.VISCOSITY_REF,\n"
    "            default_T_c: float = 250.0,\n"
    "            use_green_method: bool = True,\n"
    "            use_composite_transition_closure: bool = False,\n"
    "            d_grain: Optional[float] = None,\n"
    "            d_del: Optional[float] = None,\n"
    "            D0v: Optional[float] = None,\n"
    "            D0b: Optional[float] = None,\n"
    "            Q_b: Optional[float] = None,\n"
    "            p_grain: Optional[float] = None,\n"
    "    ) -> Tuple[float, float]:"
)
if old3 in c:
    c = c.replace(old3, new3, 1)
    changed = True
    print("  [3] Patched compute_transition_temperature signature")

# --- 4. compute_transition_temperature call to green_cond ---
old4 = (
    "                T_c, Ti = IceConvection.green_cond_base_temp(\n"
    "                    T_melt, T_surface, Q_v, eta_ref\n"
    "                )"
)
new4 = (
    "                T_c, Ti = IceConvection.green_cond_base_temp(\n"
    "                    T_melt, T_surface, Q_v, eta_ref,\n"
    "                    use_composite_transition_closure=use_composite_transition_closure,\n"
    "                    d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,\n"
    "                    Q_b=Q_b, p_grain=p_grain\n"
    "                )"
)
if old4 in c:
    c = c.replace(old4, new4, 1)
    changed = True
    print("  [4] Patched compute_transition_temperature call to green_cond_base_temp")

# --- 5. scan_temperature_profile signature ---
old5 = (
    "            use_composite_viscosity: bool = True,\n"
    "            eta_ref: float = Rheology.VISCOSITY_REF,\n"
    "    ) -> ConvectionState:"
)
new5 = (
    "            use_composite_viscosity: bool = True,\n"
    "            eta_ref: float = Rheology.VISCOSITY_REF,\n"
    "            use_composite_transition_closure: bool = False,\n"
    "            d_del: Optional[float] = None,\n"
    "            D0v: Optional[float] = None,\n"
    "            D0b: Optional[float] = None,\n"
    "    ) -> ConvectionState:"
)
if old5 in c:
    c = c.replace(old5, new5, 1)
    changed = True
    print("  [5] Patched scan_temperature_profile signature")

# --- 6. scan_temperature_profile call to compute_transition ---
old6 = (
    "        T_c, Ti = IceConvection.compute_transition_temperature(\n"
    "            T_melt, T_surface, Q_v, eta_ref, use_green_method=ConvectionConstants.USE_GREEN_METHOD\n"
    "        )"
)
new6 = (
    "        T_c, Ti = IceConvection.compute_transition_temperature(\n"
    "            T_melt, T_surface, Q_v, eta_ref, use_green_method=ConvectionConstants.USE_GREEN_METHOD,\n"
    "            use_composite_transition_closure=use_composite_transition_closure,\n"
    "            d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,\n"
    "            Q_b=Q_b, p_grain=p_grain\n"
    "        )"
)
if old6 in c:
    c = c.replace(old6, new6, 1)
    changed = True
    print("  [6] Patched scan_temperature_profile call to compute_transition_temperature")

# --- 7. is_convecting signature ---
old7 = (
    "            d_grain: Optional[float] = None,\n"
    "            p_grain: Optional[float] = None,\n"
    "    ) -> Tuple[bool, float]:"
)
new7 = (
    "            d_grain: Optional[float] = None,\n"
    "            p_grain: Optional[float] = None,\n"
    "            use_composite_transition_closure: bool = False,\n"
    "            d_del: Optional[float] = None,\n"
    "            D0v: Optional[float] = None,\n"
    "            D0b: Optional[float] = None,\n"
    "    ) -> Tuple[bool, float]:"
)
if old7 in c:
    c = c.replace(old7, new7, 1)
    changed = True
    print("  [7] Patched is_convecting signature")

# --- 8. is_convecting call to compute_transition ---
old8 = (
    "        T_c, Ti = IceConvection.compute_transition_temperature(\n"
    "            T_melt, T_surface, Q_v, eta_ref, use_green_method=ConvectionConstants.USE_GREEN_METHOD\n"
    "        )"
)
new8 = (
    "        T_c, Ti = IceConvection.compute_transition_temperature(\n"
    "            T_melt, T_surface, Q_v, eta_ref, use_green_method=ConvectionConstants.USE_GREEN_METHOD,\n"
    "            use_composite_transition_closure=use_composite_transition_closure,\n"
    "            d_grain=d_grain, d_del=d_del, D0v=D0v, D0b=D0b,\n"
    "            Q_b=Q_b, p_grain=p_grain\n"
    "        )"
)
if old8 in c:
    c = c.replace(old8, new8, 1)
    changed = True
    print("  [8] Patched is_convecting call to compute_transition_temperature")


if changed:
    p.write_text(c, encoding="utf-8")
    print("\nAll patches applied successfully to Convection.py.")
else:
    print("\nNo patches applied!")
