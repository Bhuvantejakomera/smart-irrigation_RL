import os
import matplotlib
matplotlib.use('Agg')
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC
from irrigation_env import FAO56IrrigationEnv, RuleBasedAgronomicAgent, get_env


def train(agent_type: str, reward_type: str, total_timesteps: int = 100000):
    print(f"🚀 Training {agent_type} agent with '{reward_type}' reward ({total_timesteps} timesteps)...")

    env = get_env(reward_type)

    if agent_type == "PPO":
        # Hyperparameter tuned PPO for continuous action space exploration
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.015,  # Encourages continuous action entropy exploration
            verbose=0,
            seed=42,
        )
    elif agent_type == "SAC":
        # Off-policy Soft Actor-Critic with automatic entropy tuning
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=128,
            ent_coef="auto",
            gamma=0.99,
            tau=0.005,
            verbose=0,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


def evaluate(model, reward_type: str, seed: int = 42):
    env = FAO56IrrigationEnv(reward_type=reward_type)
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    total_water = 0.0
    total_runoff = 0.0
    
    wp_series = []
    actions = []
    stages = []
    moisture_series = []
    biomass_series = []
    ks_series = []

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        irrigation = float(np.clip(action[0], 0.0, 40.0))
        total_water += irrigation

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        total_runoff += info.get("runoff", 0.0)

        current_biomass = obs[2]
        wp_step = current_biomass / (total_water + 1e-6)
        
        wp_series.append(wp_step)
        actions.append(irrigation)
        stages.append(obs[1])
        moisture_series.append(obs[0])
        biomass_series.append(current_biomass)
        ks_series.append(obs[4])

    final_biomass = obs[2]
    final_wp = final_biomass / (total_water + 1e-6)

    return {
        "total_reward": total_reward,
        "total_water": total_water,
        "final_biomass": final_biomass,
        "final_wp": final_wp,
        "total_runoff": total_runoff,
        "mean_ks": float(np.mean(ks_series)),
        "actions": actions,
        "stages": stages,
        "moisture_series": moisture_series,
        "biomass_series": biomass_series,
        "wp_series": wp_series,
    }


def run_experiments():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Experiments suite including Rule-Based baseline
    experiments = [
        ("E0", "RuleBased", "dynamic"),
        ("E1", "PPO", "fixed"),
        ("E2", "PPO", "dynamic"),
        ("E3", "SAC", "fixed"),
        ("E4", "SAC", "dynamic"),
    ]

    eval_data = {}
    summary_results = {}

    for exp, agent, reward in experiments:
        if agent == "RuleBased":
            print(f"🌾 Evaluating Rule-Based Agronomic Baseline Agent (E0)...")
            model = RuleBasedAgronomicAgent()
        else:
            model = train(agent, reward, total_timesteps=40000)
            model_path = f"models/{exp}_{agent}_{reward}"
            model.save(model_path)
            print(f"   Saved model to {model_path}.zip")

        eval_res = evaluate(model, reward, seed=42)
        eval_data[exp] = eval_res
        
        summary_results[exp] = (
            agent,
            reward,
            eval_res["total_reward"],
            eval_res["total_water"],
            eval_res["final_biomass"],
            eval_res["final_wp"],
            eval_res["total_runoff"],
            eval_res["mean_ks"],
        )

    # ==============================
    # 📋 Print & Export Table
    # ==============================
    print("\n" + "=" * 80)
    print("           🌾 REAL FAO-56 SMART IRRIGATION BENCHMARK RESULTS")
    print("=" * 80)
    header = f"{'Exp':<5} {'Agent':<10} {'Reward':<9} {'Reward':<10} {'Water (mm)':<12} {'Yield':<9} {'WP (g/mm)':<12} {'Stress Index':<12}"
    print(header)
    print("-" * 80)

    csv_rows = [["Experiment", "Agent", "Reward_Type", "Total_Reward", "Water_Used_mm", "Yield_Biomass", "Water_Productivity", "Runoff_mm", "Stress_Index"]]

    for exp, data in summary_results.items():
        agent, reward, r, w, y, wp, runoff, ks = data
        row_str = f"{exp:<5} {agent:<10} {reward:<9} {r:<10.2f} {w:<12.2f} {y:<9.2f} {wp:<12.4f} {ks:<12.2f}"
        print(row_str)
        csv_rows.append([exp, agent, reward, f"{r:.2f}", f"{w:.2f}", f"{y:.2f}", f"{wp:.4f}", f"{runoff:.2f}", f"{ks:.2f}"])

    print("=" * 80 + "\n")

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print("Saved results to results.csv")

    # Export json for dashboard consumption
    dashboard_json = {}
    for exp, res in eval_data.items():
        dashboard_json[exp] = {
            "agent": str(summary_results[exp][0]),
            "reward_type": str(summary_results[exp][1]),
            "total_reward": float(summary_results[exp][2]),
            "total_water": float(summary_results[exp][3]),
            "yield": float(summary_results[exp][4]),
            "wp": float(summary_results[exp][5]),
            "runoff": float(summary_results[exp][6]),
            "mean_ks": float(summary_results[exp][7]),
            "actions": [float(x) for x in res["actions"]],
            "stages": [float(x) for x in res["stages"]],
            "moisture": [float(x) for x in res["moisture_series"]],
            "biomass": [float(x) for x in res["biomass_series"]],
        }
        
    with open("results.json", "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print("Exported results.json for Interactive Dashboard")

    # ==============================
    # 📈 Publication Visualizations
    # ==============================
    plt.style.use('ggplot')
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    exp_keys = list(summary_results.keys())

    # Plot 1: Total Water Applied
    plt.figure(figsize=(9, 5))
    water_vals = [summary_results[k][3] for k in exp_keys]
    bars1 = plt.bar(exp_keys, water_vals, color=colors, alpha=0.85, edgecolor='black')
    plt.title("Total Water Applied (mm) across Agents & Policies", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Water Applied (mm)", fontsize=12)
    plt.xlabel("Experiment", fontsize=12)
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f} mm", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/water_used.png", dpi=300)
    plt.close()

    # Plot 2: Water Productivity (WP)
    plt.figure(figsize=(9, 5))
    wp_vals = [summary_results[k][5] for k in exp_keys]
    bars2 = plt.bar(exp_keys, wp_vals, color=colors, alpha=0.85, edgecolor='black')
    plt.title("Water Productivity (Yield / Water Applied)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("WP Ratio (Biomass / mm Water)", fontsize=12)
    plt.xlabel("Experiment", fontsize=12)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0001, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/water_productivity.png", dpi=300)
    plt.close()

    # Plot 3: Soil Moisture Trajectories vs FAO Thresholds
    plt.figure(figsize=(10, 6))
    for idx, exp in enumerate(exp_keys):
        plt.plot(eval_data[exp]["moisture_series"], label=f"{exp} ({summary_results[exp][0]})", linewidth=2.0, color=colors[idx])
    plt.axhline(y=0.38, color='blue', linestyle='--', alpha=0.7, label='Field Capacity (FC=0.38)')
    plt.axhline(y=0.265, color='orange', linestyle='--', alpha=0.7, label='RAW Threshold (0.265)')
    plt.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Wilting Point (WP=0.15)')
    plt.title("Soil Moisture Trajectories vs FAO-56 Agronomic Thresholds", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Day (Time Step)", fontsize=12)
    plt.ylabel("Soil Moisture Content (vol/vol)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/soil_moisture_trajectories.png", dpi=300)
    plt.close()

    # Plot 4: Biomass Accumulation Over Time
    plt.figure(figsize=(10, 6))
    for idx, exp in enumerate(exp_keys):
        plt.plot(eval_data[exp]["biomass_series"], label=f"{exp} ({summary_results[exp][0]}-{summary_results[exp][1]})", linewidth=2.5, color=colors[idx])
    plt.title("Crop Biomass Accumulation Trajectory", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Day (Time Step)", fontsize=12)
    plt.ylabel("Biomass Yield Index", fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/biomass_growth.png", dpi=300)
    plt.close()

    # Plot 5: Irrigation Action Policy Curves
    plt.figure(figsize=(10, 6))
    for idx, exp in enumerate(exp_keys):
        plt.plot(eval_data[exp]["stages"], eval_data[exp]["actions"], label=f"{exp} ({summary_results[exp][0]})", linewidth=2.0, color=colors[idx])
    plt.title("Irrigation Policy Actions across Growth Stages", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Crop Growth Stage (t = day / 120)", fontsize=12)
    plt.ylabel("Irrigation Volume (mm/day)", fontsize=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/actions_vs_stage.png", dpi=300)
    plt.close()

    print("All visualizations saved in plots/ directory.")


if __name__ == "__main__":
    run_experiments()
