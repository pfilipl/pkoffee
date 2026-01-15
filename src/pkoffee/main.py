import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import seaborn as sns


class Model:
    def __init__(self, name, p0, bounds, function):
        self.Name = name
        self.P0 = p0
        self.Bounds = bounds
        self.Function = function

    def fit(self, X, Y):
        popt, _ = curve_fit(
            self.Function, X, Y, p0=self.P0, bounds=self.Bounds, maxfev=20000
        )
        yhat = self.Function(X, *popt)
        ss_res = float(np.sum((Y - yhat) ** 2))
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"name": self.Name, "func": self.Function, "params": popt, "r2": r2}


# Reading data
data = pd.read_csv("coffee_productivity.csv")
X = data["cups"].values
Y = data["productivity"].values

# Configurate plot
fig = plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_xlabel("Cups of Coffee")
ax.set_ylabel("Productivity")
ax.set_title("Productivity vs Coffee")
ax.grid(True, alpha=0.3)

sns.violinplot(
    data=data,
    x="cups",
    y="productivity",
    hue="cups",
    ax=ax,
    inner="quartile",
    cut=0,
    density_norm="width",
    palette="Greens",
    linewidth=0.8,
    legend=False,
)

x_min, x_max = float(np.min(X)), float(np.max(X))
y_min, y_max = float(np.min(Y)), float(np.max(Y))
dy = max(1e-8, y_max - y_min)

# Models definitions
MODELS = [
    Model(
        "quadratic",
        [y_min, 0.0, 0.01],
        (-np.inf, np.inf),
        lambda xx, a0, a1, a2: a0 + a1 * xx + a2 * xx**2,
    ),
    Model(
        "saturating",
        [dy, max(1.0, 0.2 * (x_min + x_max)), y_min],
        ([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        lambda xx, Vmax, K, y0: y0 + Vmax * (xx / np.maximum(K + xx, 1e-9)),
    ),
    Model(
        "logistic",
        [dy, 0.5, 0.5 * (x_min + x_max), y_min],
        ([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        lambda xx, L, k, x0, y0: y0 + L / (1.0 + np.exp(-k * (xx - x0))),
    ),
    Model(
        "peak",
        [max(y_min, y_max), max(1.0, 0.5 * (x_min + x_max))],
        ([-np.inf, 0.0], [np.inf, np.inf]),
        lambda xx, a, b: a * xx * np.exp(-xx / np.maximum(b, 1e-9)),
    ),
    Model(
        "peak2",
        [max(1e-6, y_max / max(1.0, x_max**2)), max(1.0, 0.5 * (x_min + x_max))],
        ([-np.inf, 0.0], [np.inf, np.inf]),
        lambda xx, a, b: a * (xx**2) * np.exp(-xx / np.maximum(b, 1e-9)),
    ),
]

# Fitting
fits = []
for m in MODELS:
    fits.append(m.fit(X, Y))

fits.sort(key=lambda d: (d["r2"] if np.isfinite(d["r2"]) else -np.inf), reverse=True)

# Scoring
x_smooth = np.linspace(np.min(X), np.max(X), 300)
for idx, res in enumerate(fits):
    y_s = res["func"](x_smooth, *res["params"])
    label = f"{res['name']} (R²={res['r2']:.3f})"
    ax.plot(x_smooth, y_s, lw=2, label=label)

# Plotting
ax.legend()
ax.set_ylim(-0.2, 8)
out_path = Path(__file__).with_name("fit_plot.png")
plt.tight_layout()
# plt.savefig(out_path, dpi=150)
plt.show()
