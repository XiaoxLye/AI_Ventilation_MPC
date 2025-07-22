"""
scripts/control/mpc_controller.py
--------------------------------
Minimal MPC for indoor CO₂ control using GEKKO.

Assumptions
-----------
- Single-zone well-mixed room.
- Ventilation rate v(t) is the manipulated variable, 0 ≤ v ≤ 1 (fraction of max flow).
- Disturbances: predicted occupancy (people) & outdoor CO₂.
- 1‑min control interval, 30‑min prediction horizon (N = 30).
Dependencies
------------
pip install gekko numpy
"""

from gekko import GEKKO
import numpy as np

# -----------------------------------------------------------------------------
# User‑configurable parameters
# -----------------------------------------------------------------------------
V_ROOM   = 100.0        # m³, room volume
Q_MAX    = 300.0 / 3600 # m³/s, design ventilation (300 m³/h)
G_CO2    = 0.000005     # m³/s, CO₂ generation per person (~0.3 L/min)
CO2_SET  = 900.0        # ppm, comfort set‑point
C_INIT   = 1100.0       # ppm, initial indoor CO₂
LAMBDA_U = 1e-3        # weight on control effort
DT       = 60           # s, time step
N_HORIZ  = 30           # 30 steps × 60 s = 30 min

# -----------------------------------------------------------------------------
def mpc_step(c_now, occ_pred, co2_out_pred):
    """
    Run one MPC optimisation and return the first ventilation action.

    Parameters
    ----------
    c_now : float
        Current indoor CO₂ concentration (ppm).
    occ_pred : array‑like (N_HORIZ,)
        Predicted occupancy (# persons) for next horizon.
    co2_out_pred : array‑like (N_HORIZ,)
        Predicted outdoor CO₂ (ppm).
    """
    m = GEKKO(remote=False)          # local solve

    # time grid
    m.time = np.arange(N_HORIZ + 1)

    # variables
    c = m.Var(value=c_now)           # indoor CO₂ ppm
    v = m.MV(lb=0, ub=1, value=0.3)  # ventilation fraction, MV

    v.STATUS = 1                     # allow optimizer to change
    v.DCOST  = 0                     # no move suppression

    # parameters / disturbances
    occ   = m.Param(value=occ_pred)
    co2_o = m.Param(value=co2_out_pred)

    # model differential eq (discretised as GEKKO ODE)
    # dc/dt = (Q/V)*(c_out ‑ c) + G/V
    m.Equation(c.dt() == (Q_MAX * v / V_ROOM) * (co2_o - c) +
               (G_CO2 * occ) / V_ROOM * 1e6)   # convert m³/s→ppm/s

    # objective: track set‑point & penalise ventilation effort
    m.Minimize((c - CO2_SET) ** 2)
    m.Minimize(LAMBDA_U * v ** 2)

    # options
    m.options.IMODE = 6   # MPC
    m.options.NODES = 2
    m.options.SOLVER = 1  # APOPT
    m.solve(disp=False)

    return float(v.NEWVAL)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # dummy predictions for demo
    occ_pred      = np.ones(N_HORIZ + 1) * 2          # 2 persons
    co2_out_pred  = np.ones(N_HORIZ + 1) * 420        # ppm
    first_action  = mpc_step(C_INIT, occ_pred, co2_out_pred)
    print(f"Recommended ventilation fraction for next minute: {first_action:.2f}")
