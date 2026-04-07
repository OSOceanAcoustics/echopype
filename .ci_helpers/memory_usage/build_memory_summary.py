import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path("memory_artifacts")
files = list(root.glob("**/memory_usage.csv"))

groups = {
    "integration": [],
    "unit": [],
    "windows": [],
}

linestyles = {
    "3.11": "-",
    "3.12": "--",
    "3.13": ":",
}

colors = {
    "integration": "orange",
    "unit": "brown",
    "windows": "purple",
}


def parse_label(name: str):
    label = name.replace("memory-", "")

    if label.startswith("integration-"):
        group = "integration"
        pyver = label.split("-")[1]
    elif label.startswith("unit-"):
        group = "unit"
        pyver = label.split("-")[1]
    elif label.startswith("windows-"):
        group = "windows"
        pyver = label.split("-")[1]
    else:
        return None, None

    return group, pyver


all_x = []

for f in sorted(files):
    label = f.parent.name
    group, pyver = parse_label(label)
    if group is None:
        continue

    times = []
    used = []

    with f.open() as fh:
        reader = csv.DictReader(fh)
        t0 = None
        for row in reader:
            t = float(row["timestamp"])
            if t0 is None:
                t0 = t
            times.append((t - t0) / 60.0)
            used.append(float(row["mem_used_mb"]))

    if times:
        all_x.extend(times)
        groups[group].append((pyver, times, used))

xmax = max(all_x) if all_x else 1

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

subplot_order = [
    ("integration", "Ubuntu integration"),
    ("unit", "Ubuntu unit"),
    ("windows", "Windows"),
]

for ax, (group, title) in zip(axes, subplot_order):
    for pyver, times, used in sorted(groups[group], key=lambda x: x[0]):
        ax.plot(
            times,
            used,
            label=pyver,
            color=colors[group],
            linestyle=linestyles.get(pyver, "-"),
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_ylabel("RAM used (MB)")
    ax.set_xlim(0, xmax)
    ax.legend(title="Python")

axes[-1].set_xlabel("Time since job start (minutes)")

fig.suptitle("CI memory usage by job", fontsize=16)
fig.tight_layout()
fig.savefig("memory_all_jobs.png", dpi=150)
