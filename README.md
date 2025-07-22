# AI_Ventilation_MPC
AI-enhanced ventilation control system integrating IAQ prediction (CO₂/PM2.5) and MPC.
# AI-Enhanced Ventilation Control System

An open-source research framework for **residential ventilation control** that combines:

- ML-based indoor air quality (IAQ) prediction (CO₂, PM2.5).
- Occupancy & window-state forecasting.
- Weather forecast integration.
- **Model Predictive Control (MPC)** to coordinate natural + mechanical ventilation for IAQ + energy performance.

---

## Project Structure


## Repo Layout
- `data/` raw & processed sensor + weather + occupancy data
- `models/iaq_prediction/` ML models (LSTM, XGBoost)
- `models/mpc/` optimization models (GEKKO / CasADi)
- `scripts/` preprocessing, training, control loop runners
- `notebooks/` exploratory analysis & experiments
- `config/` parameter & system configuration
- `results/` experiment logs, plots

## Quick Start
```bash
pip install -r requirements.txt
python main.py
