import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path("memory_artifacts")
files = list(root.glob("**/memory_usage.csv"))

plt.figure(figsize=(12, 5))

for f in sorted(files):
    label = f.parent.name.replace("memory-", "")
    times = []
    used = []

    with f.open() as fh:
        reader = csv.DictReader(fh)
        t0 = None

        for row in reader:
            t = float(row["timestamp"])

            if t0 is None:
                t0 = t

            times.append((t - t0) / 60)
            used.append(float(row["mem_used_mb"]))

    linestyle = "--" if "windows" in label.lower() else "-"
    plt.plot(times, used, label=label, linestyle=linestyle)

plt.xlabel("Time since job start (minutes)")
plt.ylabel("RAM used (MB)")
plt.title("CI memory usage by job")
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig("memory_all_jobs.png", dpi=150)
