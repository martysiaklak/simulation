import pandas as pd
import matplotlib.pyplot as plt

# ---------- Load monitoring results ----------
df = pd.read_csv("monitor_results.csv")

# ---------- Add smoothed columns ----------
window = 10  # розмір вікна для ковзного середнього (можеш змінити)
for col in ["PrepQueue", "TheatreQueue", "RecoveryQueue",
            "PrepUtil", "TheatreUtil", "RecoveryUtil"]:
    df[f"{col}_Smooth"] = df[col].rolling(window=window, min_periods=1).mean()

# ---------- Queue length plots ----------
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["PrepQueue"], color="tab:blue", alpha=0.3, label="Preparation Queue (raw)")
plt.plot(df["Time"], df["TheatreQueue"], color="tab:orange", alpha=0.3, label="Theatre Queue (raw)")
plt.plot(df["Time"], df["RecoveryQueue"], color="tab:green", alpha=0.3, label="Recovery Queue (raw)")

# Плавні лінії поверх сирих
plt.plot(df["Time"], df["PrepQueue_Smooth"], color="tab:blue", linewidth=2.5, label="Preparation Queue (smoothed)")
plt.plot(df["Time"], df["TheatreQueue_Smooth"], color="tab:orange", linewidth=2.5, label="Theatre Queue (smoothed)")
plt.plot(df["Time"], df["RecoveryQueue_Smooth"], color="tab:green", linewidth=2.5, label="Recovery Queue (smoothed)")

plt.xlabel("Time")
plt.ylabel("Queue length")
plt.title("Queue Lengths Over Time (Smoothed)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("queue_lengths_smoothed.png", dpi=300)
plt.show()


# ---------- Utilisation plots ----------
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], 100 * df["PrepUtil"], color="tab:blue", alpha=0.3, label="Preparation Utilisation (raw)")
plt.plot(df["Time"], 100 * df["TheatreUtil"], color="tab:orange", alpha=0.3, label="Theatre Utilisation (raw)")
plt.plot(df["Time"], 100 * df["RecoveryUtil"], color="tab:green", alpha=0.3, label="Recovery Utilisation (raw)")

# Плавні лінії
plt.plot(df["Time"], 100 * df["PrepUtil_Smooth"], color="tab:blue", linewidth=2.5, label="Preparation Utilisation (smoothed)")
plt.plot(df["Time"], 100 * df["TheatreUtil_Smooth"], color="tab:orange", linewidth=2.5, label="Theatre Utilisation (smoothed)")
plt.plot(df["Time"], 100 * df["RecoveryUtil_Smooth"], color="tab:green", linewidth=2.5, label="Recovery Utilisation (smoothed)")

plt.xlabel("Time")
plt.ylabel("Utilisation (%)")
plt.title("Resource Utilisation Over Time (Smoothed)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("utilisation_smoothed.png", dpi=300)
plt.show()

print("\n Smoothed plots saved as 'queue_lengths_smoothed.png' and 'utilisation_smoothed.png'")
