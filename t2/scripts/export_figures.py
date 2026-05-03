"""Generate the 8 figures referenced in relatorio_final.tex.

Figures 1-5 use only data/training_results.csv.
Figures 6-8 require a quick retrain of the best agent (SARSA linear, alpha=1e-4, gamma=0.9)
plus a Value Iteration run.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from env.env import StockMarketEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.value_iteration import ValueIterationAgent

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 150

TRAIN_CSV = os.path.join(ROOT, "data", "amzn_transformed_train.csv")
TEST_CSV = os.path.join(ROOT, "data", "amzn_transformed_test.csv")
RESULTS_CSV = os.path.join(ROOT, "data", "training_results.csv")

LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
GAMMA_VALUES = [0.3, 0.5, 0.8, 0.9]
EPSILON_DECAY_TYPES = ["constant", "linear", "exponential"]
AGENT_TYPES = ["Q-Learning"]
SMOOTH_WINDOW = 50

print("Loading training_results.csv ...")
df = pd.read_csv(RESULTS_CSV)


def smooth(s):
    return pd.Series(s).rolling(window=SMOOTH_WINDOW, min_periods=1).mean()


def get_rewards(agent_type, decay, lr, gamma):
    sub = df[(df.agent_type == agent_type) & (df.decay_type == decay)
             & (df.learning_rate == lr) & (df.gamma == gamma)]
    return sub.sort_values("episode").reward.values


def get_epsilons(agent_type, decay, lr, gamma):
    sub = df[(df.agent_type == agent_type) & (df.decay_type == decay)
             & (df.learning_rate == lr) & (df.gamma == gamma)]
    return sub.sort_values("episode").epsilon.values


# =============================================================
# Fig 2: Learning curves by decay type (gamma=0.8 panel)
# =============================================================
print("[2/8] learning_curves_by_decay.png")
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(LEARNING_RATES)))
line_styles = {"Q-Learning": "-", "SARSA": "--"}
gamma = 0.8
for ax_idx, decay_type in enumerate(EPSILON_DECAY_TYPES):
    ax = axes[ax_idx]
    ax.set_title(f"ε-decay: {decay_type}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episódio")
    if ax_idx == 0:
        ax.set_ylabel("Reward Acumulado (média móvel)")
    for agent_type in AGENT_TYPES:
        for lr_idx, lr in enumerate(LEARNING_RATES):
            r = get_rewards(agent_type, decay_type, lr, gamma)
            if len(r) == 0:
                continue
            ax.plot(smooth(r), label=f"{agent_type} (α={lr})",
                    color=colors[lr_idx], linestyle=line_styles[agent_type],
                    linewidth=1.5, alpha=0.85)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle(f"Curvas de Aprendizado — Reward Acumulado por Episódio (Q-Learning, γ={gamma})",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "learning_curves_by_decay_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Fig 3: Epsilon decay
# =============================================================
print("[3/8] epsilon_decay.png")
decay_colors = {"constant": "#e74c3c", "linear": "#2ecc71", "exponential": "#3498db"}
fig, ax = plt.subplots(figsize=(10, 5))
for decay_type in EPSILON_DECAY_TYPES:
    eps = get_epsilons("Q-Learning", decay_type, LEARNING_RATES[0], GAMMA_VALUES[0])
    if len(eps):
        ax.plot(eps, label=decay_type, color=decay_colors[decay_type], linewidth=2)
ax.set_xlabel("Episódio", fontsize=12)
ax.set_ylabel("Epsilon (ε)", fontsize=12)
ax.set_title("Evolução do Epsilon por Estratégia de Decaimento", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(-0.02, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "epsilon_decay_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Fig 4: Gamma impact (curve + boxplot)
# =============================================================
print("[4/8] gamma_impact.png")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
agent = "Q-Learning"
decay_type = "linear"
lr = 0.0001
gamma_colors = {0.3: "#e74c3c", 0.5: "#f1c40f", 0.8: "#e67e22", 0.9: "#2ecc71"}
for gv in GAMMA_VALUES:
    r = get_rewards(agent, decay_type, lr, gv)
    if len(r) == 0:
        continue
    axes[0].plot(smooth(r), label=f"γ = {gv}", color=gamma_colors[gv], linewidth=2)
    axes[1].boxplot(r[-100:], positions=[GAMMA_VALUES.index(gv)], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor=gamma_colors[gv], alpha=0.6))
axes[0].set_title(f"Impacto do Fator de Desconto (γ) na Recompensa\n({agent}, {decay_type}, α={lr})",
                  fontweight="bold")
axes[0].set_xlabel("Episódio")
axes[0].set_ylabel("Reward Acumulado (Média Móvel)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].set_title("Estabilidade nos Últimos 100 Episódios", fontweight="bold")
axes[1].set_xticks(range(len(GAMMA_VALUES)))
axes[1].set_xticklabels([f"γ = {g}" for g in GAMMA_VALUES])
axes[1].set_ylabel("Reward")
axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "gamma_impact_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Fig 5: Heatmap decay × alpha
# =============================================================
print("[5/8] heatmap_decay_alpha.png")
gamma = 0.8
fig, axes = plt.subplots(1, len(AGENT_TYPES), figsize=(8 * len(AGENT_TYPES), 5), squeeze=False)
axes = axes[0]
for ax_idx, agent_type in enumerate(AGENT_TYPES):
    ax = axes[ax_idx]
    heatmap_data = np.zeros((len(EPSILON_DECAY_TYPES), len(LEARNING_RATES)))
    for i, dec in enumerate(EPSILON_DECAY_TYPES):
        for j, lr in enumerate(LEARNING_RATES):
            r = get_rewards(agent_type, dec, lr, gamma)
            heatmap_data[i, j] = np.mean(r[-100:]) if len(r) else np.nan
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(LEARNING_RATES)))
    ax.set_xticklabels([f"α={lr}" for lr in LEARNING_RATES])
    ax.set_yticks(range(len(EPSILON_DECAY_TYPES)))
    ax.set_yticklabels(EPSILON_DECAY_TYPES)
    for i in range(len(EPSILON_DECAY_TYPES)):
        for j in range(len(LEARNING_RATES)):
            ax.text(j, i, f"{heatmap_data[i, j]:.4f}", ha="center", va="center",
                    fontsize=11, fontweight="bold")
    ax.set_title(f"{agent_type} (γ = {gamma})", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Reward Médio (últimos 100 eps)")
plt.suptitle("Heatmap de Recompensas: Estratégias de Exploração vs Learning Rate",
             fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "heatmap_decay_alpha_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Heavy: train best SARSA agent for V(s), policy and trajectory
# =============================================================
BEST_LR = 1e-4
BEST_GAMMA = 0.3
BEST_DECAY = "exponential"
BEST_LABEL = f"Q-Learning ({BEST_DECAY}, α={BEST_LR}, γ={BEST_GAMMA})"
print(f"Training best Q-Learning agent ({BEST_DECAY}, α={BEST_LR}, γ={BEST_GAMMA}, 5000 eps) ...")
np.random.seed(42)
env = StockMarketEnv(TRAIN_CSV)
best = QLearningAgent(
    env=env, learning_rate=BEST_LR, gamma=BEST_GAMMA,
    epsilon_decay_type=BEST_DECAY,
)
N_EPISODES = 5000
for ep in range(N_EPISODES):
    state = env.reset()
    done = False
    while not done:
        action = best.choose_action(state, training=True)
        next_state, reward, done = env.step(action)
        best.update(state, action, reward, next_state, done)
        state = next_state
    best.update_epsilon(ep)
    if ep % 500 == 0:
        print(f"  ep={ep}/{N_EPISODES} ε={best.epsilon:.3f}")

# =============================================================
# Fig 6: V(s) and policy maps
# =============================================================
print("[6/8] value_policy.png")
v_map = np.zeros((4, 4))
policy_map = np.zeros((4, 4))
action_colors = {0: "#e74c3c", 1: "#95a5a6", 2: "#2ecc71"}
action_labels = {0: "Sell", 1: "Hold", 2: "Buy"}
axis_labels = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
for s0 in range(2):
    for s1 in range(2):
        for s2 in range(2):
            for s3 in range(2):
                row = s0 * 2 + s1
                col = s2 * 2 + s3
                qv = best.q_table[s0, s1, s2, s3]
                v_map[row, col] = np.max(qv)
                policy_map[row, col] = np.argmax(qv)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im_v = axes[0].imshow(v_map, cmap="viridis")
axes[0].set_title("Mapa de Valores V(s)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Estados (μ, ρ)")
axes[0].set_ylabel("Estados (p, υ)")
axes[0].set_xticks(range(4)); axes[0].set_yticks(range(4))
axes[0].set_xticklabels(axis_labels); axes[0].set_yticklabels(axis_labels)
for i in range(4):
    for j in range(4):
        text_color = "black" if v_map[i, j] > np.max(v_map) * 0.7 else "white"
        axes[0].text(j, i, f"{v_map[i, j]:.2f}", ha="center", va="center",
                     color=text_color, fontweight="bold")
fig.colorbar(im_v, ax=axes[0], label="Valor Esperado Máximo")

policy_rgb = np.zeros((4, 4, 3))
for i in range(4):
    for j in range(4):
        ai = int(policy_map[i, j])
        hx = action_colors[ai].lstrip('#')
        policy_rgb[i, j] = tuple(int(hx[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
axes[1].imshow(policy_rgb)
axes[1].set_title("Política Aprendida π(s)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Estados (μ, ρ)")
axes[1].set_ylabel("Estados (p, υ)")
axes[1].set_xticks(range(4)); axes[1].set_yticks(range(4))
axes[1].set_xticklabels(axis_labels); axes[1].set_yticklabels(axis_labels)
for i in range(4):
    for j in range(4):
        ai = int(policy_map[i, j])
        axes[1].text(j, i, action_labels[ai], ha="center", va="center",
                     color="white", fontweight="bold")
plt.suptitle(f"Análise do Melhor Agente: {BEST_LABEL}",
             fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "value_policy_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Evaluate best agent on test set (greedy)
# =============================================================
print("Evaluating best agent on test set ...")
test_env = StockMarketEnv(TEST_CSV)
state = test_env.reset()
test_rewards = []
positions = []
done = False
while not done:
    action = best.choose_action(state, training=False)
    next_state, r, done = test_env.step(action)
    test_rewards.append(r)
    positions.append(test_env.position)
    state = next_state
print(f"  total test reward = {sum(test_rewards):.4f}")

# =============================================================
# Fig 7: Train vs Test (top configs by training)
# =============================================================
print("[7/8] train_vs_test.png — building from training_results.csv top configs (test reward = best agent)")
last100 = (df[df.episode >= 4900]
           .groupby(['agent_type', 'decay_type', 'learning_rate', 'gamma'])
           .reward.mean()
           .reset_index()
           .sort_values('reward', ascending=False))
top10 = last100.head(10).copy()
# We only have measured test reward for the single best agent; fill rest with NaN-like 0 + a marker.
# For a useful plot, train test reward only for the best (already done) and show it as a marker; show train bars for top10.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
labels = [f"{r.agent_type}\n{r.decay_type}\nα={r.learning_rate}\nγ={r.gamma}"
          for r in top10.itertuples()]
x = np.arange(len(labels)); width = 0.6
axes[0].bar(x, top10.reward.values, width, color="#3498db", alpha=0.85,
            label="Treino (média últimos 100 ep)")
# Highlight test reward of the actual best (row 0 of top10 should match SARSA linear 1e-4 0.9)
best_match = (top10.iloc[0].agent_type == "Q-Learning"
              and top10.iloc[0].decay_type == BEST_DECAY
              and abs(top10.iloc[0].learning_rate - BEST_LR) < 1e-9
              and abs(top10.iloc[0].gamma - BEST_GAMMA) < 1e-9)
test_total = sum(test_rewards)
axes[0].scatter([0 if best_match else None], [test_total], color="#e74c3c", s=120, zorder=5,
                label=f"Teste total (melhor agente): {test_total:.2f}")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
axes[0].set_ylabel("Reward")
axes[0].set_title("Treino (Top 10 Configurações) e Reward de Teste do Melhor Agente",
                  fontweight="bold")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

cumulative = np.cumsum(test_rewards)
axes[1].plot(cumulative, color="#2ecc71", linewidth=1.5)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Reward Acumulado")
axes[1].set_title(f"Retorno Acumulado no Teste ({BEST_LABEL})",
                  fontweight="bold")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "train_vs_test_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Fig 8: Trajectory
# =============================================================
print("[8/8] trajectory.png")
test_prices = pd.read_csv(TEST_CSV)["Close"].values
pos_array = np.array(positions)
fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.plot(test_prices[1:len(pos_array) + 1], color="#2c3e50", linewidth=1, label="Preço Close")
ax1.set_xlabel("Step")
ax1.set_ylabel("Preço (Close)", color="#2c3e50")
ymin, ymax = ax1.get_ylim()
ax1.fill_between(range(len(pos_array)), ymin, ymax,
                 where=pos_array == 1, alpha=0.15, color="green", label="Comprado (long)")
ax1.legend(loc="upper left")
ax1.set_title(f"Trajetória do Agente no Teste — {BEST_LABEL}",
              fontsize=14, fontweight="bold")
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "trajectory_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()


# =============================================================
# Fig 1: Value Iteration delta history
# =============================================================
print("[1/8] vi_delta.png — running Value Iteration with history")
vi_env = StockMarketEnv(TRAIN_CSV, noise_pct=0.0)
vi_env.reset()
vi = ValueIterationAgent(vi_env, gamma=0.9, theta=1e-4, max_iterations=2000)
delta_history = []
converged_at = None
for it in range(vi.max_iterations):
    prev_V = vi.value_table.copy()
    # Sweep over all states
    for s0 in range(2):
        for s1 in range(2):
            for s2 in range(2):
                for s3 in range(2):
                    vi.update((s0, s1, s2, s3))
    delta = float(np.max(np.abs(vi.value_table - prev_V)))
    delta_history.append(delta)
    if delta < vi.theta and converged_at is None:
        converged_at = it
        break

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(delta_history, color='#3498db', linewidth=1.8)
if converged_at is not None:
    ax.axvline(converged_at, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'Convergência (it={converged_at})')
ax.set_title("Value Iteration — Δ por Iteração", fontsize=14, fontweight='bold')
ax.set_xlabel('Iteração de Bellman')
ax.set_ylabel('Δ = max_s |V_{k+1}(s) - V_k(s)|')
ax.set_yscale('log')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "vi_delta_v2.png"), dpi=DPI, bbox_inches="tight")
plt.close()

print("Done. All 8 figures written to:", FIG)
