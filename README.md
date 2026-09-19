# 🌾 Deep Reinforcement Learning for Smart Precision Irrigation (FAO-56 Dual Crop Model)

This repository implements a **Deep Reinforcement Learning (DRL)** benchmark comparing **PPO (Proximal Policy Optimization)** and **SAC (Soft Actor-Critic)** agents against a traditional **Agronomic Rule-Based Threshold Controller (FAO-56)** on a simulated 120-day crop growth environment (`FAO56IrrigationEnv`).

---

## 📌 1. Real Agronomic Physics & Environment Dynamics

### 🌿 FAO-56 Dual Crop Coefficient Model (`FAO56IrrigationEnv`)
The soil-crop system models daily hydrological balance across a 120-day cropping season:

1. **Soil Hydrology Parameters:**
   - Field Capacity ($\theta_{\text{FC}}$): $0.38$ (38% volume)
   - Wilting Point ($\theta_{\text{WP}}$): $0.15$ (15% volume)
   - Readily Available Water threshold ($\theta_{\text{CRIT}}$): $0.265$ (26.5% volume)
   - Active Root Zone Depth: $400$ mm storage depth.

2. **Crop Evapotranspiration ($ET_c$):**
   $$ET_c(t) = K_c(t) \cdot ET_0(t)$$
   where $K_c(t)$ is the FAO-56 dual crop coefficient curve ($K_{c,\text{ini}} = 0.40, K_{c,\text{mid}} = 1.15, K_{c,\text{end}} = 0.55$).

3. **Soil Moisture Update & Runoff:**
   $$\theta_{t+1} = \text{clip}\left(\theta_t + \frac{a_t + P_t - ET_a}{400}, \; \theta_{\text{WP}} \cdot 0.8, \; 0.45\right)$$
   Any moisture applied beyond Field Capacity ($\theta_{\text{FC}} = 0.38$) drains as surface runoff or deep percolation.

4. **Crop Stress Index ($K_s$):**
   - $K_s = 1.0$ when $\theta_t \ge \theta_{\text{CRIT}}$ (No moisture stress).
   - Drops linearly to $0.0$ at Wilting Point $\theta_{\text{WP}}$.
   - Biomass growth $\Delta B_t$ is directly proportional to actual $ET_a = K_s \cdot ET_c$.

---

## ⚖️ 2. Reward Functions

* **Fixed Reward (`fao_fixed_reward`):**
  $$R_{\text{fixed}} = 2.0 \cdot \Delta B_t - 0.15 \cdot a_t - 0.5 \cdot (1 - K_s)$$

* **Dynamic Stage-Aware Reward (`fao_dynamic_reward`):**
  $$R_{\text{dynamic}} = \text{stage\_weight}(t) \cdot 3.0 \cdot \Delta B_t + 0.5 \cdot \tanh\left(\frac{\Delta B_t}{a_t + 1}\right) + \text{zone\_bonus} - 0.12 \cdot a_t - 0.8 \cdot (1 - K_s)$$
  - Dynamically weights biomass gain during critical flowering/grain-filling stages ($t \approx 0.5$).
  - Provides a bonus when maintaining soil moisture within the optimal RAW zone $[0.265, 0.38]$.

---

## 📊 3. Final Benchmark Results

Below are the empirical evaluation metrics obtained across the 5 experiment configurations over 120 days:

| Exp | Agent | Reward Type | Total Reward | Water Used (mm) | Yield Biomass | Water Productivity ($\frac{\text{Yield}}{\text{Water}}$) | Stress Index ($K_s$) |
|---|---|---|---|---|---|---|---|
| **E0** | **Rule-Based Baseline** | Dynamic Threshold | $-7.94$ | $525.00$ mm | $9.65$ | $0.0184$ | $0.99$ |
| **E1** | PPO | Fixed | $-42.37$ | $0.00$ mm | $1.56$ | N/A ($a=0$) | $0.24$ |
| **E2** | **PPO** | **Dynamic** | $-50.66$ | **$169.87$ mm** | $4.50$ | **$0.0265$** (Peak Efficiency) | $0.49$ |
| **E3** | SAC | Fixed | $-53.34$ | $378.80$ mm | $7.59$ | $0.0200$ | $0.81$ |
| **E4** | **SAC** | **Dynamic** | **$-6.42$** | **$510.99$ mm** | **$9.75$** (Peak Yield) | $0.0191$ | **$1.00$** |

---

## 💡 Key Takeaways

1. **SAC Dynamic (E4) is the Best Overall Policy:**
   - Achieves the **highest yield biomass ($9.75$)** and **zero crop stress ($K_s = 1.00$)** while saving **$14.01$ mm of water** compared to traditional Rule-Based Threshold Irrigation ($510.99$ mm vs $525.00$ mm).
   - Secures the highest Total Episode Reward ($-6.42$).
2. **PPO Dynamic (E2) is the Most Water-Efficient Policy:**
   - Applies only $169.87$ mm of water (saving **355 mm** over rule-based irrigation), achieving an impressive Water Productivity of **$0.0265$ g/mm** for water-scarce arid regions.

---

## 📂 4. Repository Structure

```
DL_project/
├── irrigation_env.py      # FAO-56 dual crop environment & RuleBasedAgronomicAgent class
├── train_and_evaluate.py  # Benchmark runner, model training, evaluation & plot generator
├── dashboard.html         # Interactive single-page web dashboard visualizer
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
├── results.csv            # Exported evaluation metrics CSV
├── results.json           # JSON export consumed by dashboard.html
└── plots/                 # Saved PNG charts (moisture, biomass, actions, water used)
```

---

## 🚀 5. How to Run

### Step 1: Run Training & Benchmark
```bash
/Users/srinivasa/Desktop/Projects/DL_project/venv/bin/python train_and_evaluate.py
```

### Step 2: Open Interactive Visual Dashboard
Open [dashboard.html](file:///Users/srinivasa/Desktop/Projects/DL_project/dashboard.html) in your browser!
