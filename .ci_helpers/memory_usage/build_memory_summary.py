import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path("memory_artifacts")
files = list(root.glob("**/memory_usage.csv"))

groups = {
    "ubuntu-integration": [],
    "ubuntu-unit": [],
    "windows-integration": [],
    "windows-unit": [],
}

PYTHON_LINE_STYLES = {
    "3.12": "-",
    "3.13": "--",
    "3.14": ":",
}

colors = {
    "ubuntu-integration": "orange",
    "ubuntu-unit": "brown",
    "windows-integration": "purple",
    "windows-unit": "teal",
}


def parse_label(name: str):
    label = name.removeprefix("memory-")
    parts = label.split("-")

    if parts[0] == "integration":
        return "ubuntu-integration", parts[1]

    if parts[0] == "unit":
        return "ubuntu-unit", parts[1]

    if parts[0] == "windows" and len(parts) >= 3:
        job_type = parts[1]
        pyver = parts[2]
        return f"windows-{job_type}", pyver

    return None, None


all_x = []

for file in sorted(files):
    group, pyver = parse_label(file.parent.name)

    if group not in groups:
        continue

    times = []
    used = []

    with file.open() as stream:
        reader = csv.DictReader(stream)
        t0 = None

        for row in reader:
            timestamp = float(row["timestamp"])

            if t0 is None:
                t0 = timestamp

            times.append((timestamp - t0) / 60)
            used.append(float(row["mem_used_mb"]))

    if times:
        all_x.extend(times)
        groups[group].append((pyver, times, used))

xmax = max(all_x, default=1)

subplot_order = [
    ("ubuntu-integration", "Ubuntu integration"),
    ("ubuntu-unit", "Ubuntu unit"),
    ("windows-integration", "Windows integration"),
    ("windows-unit", "Windows unit"),
]

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 13),
    sharex=True,
    sharey=True,
)

for ax, (group, title) in zip(axes, subplot_order):
    for pyver, times, used in sorted(groups[group]):
        ax.plot(
            times,
            used,
            label=pyver,
            color=colors[group],
            linestyle=PYTHON_LINE_STYLES.get(pyver, "-"),
            linewidth=2,
        )

    ax.set_title(title)
    ax.set_ylabel("RAM used (MB)")
    ax.set_xlim(0, xmax)
    ax.legend(title="Python")
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Time since job start (minutes)")

fig.suptitle("CI memory usage by job", fontsize=16)
fig.tight_layout()
fig.savefig(
    "memory_all_jobs.png",
    dpi=150,
    bbox_inches="tight",
)
