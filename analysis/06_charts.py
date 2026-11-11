import duckdb
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"     # chart background
INK     = "#0b0b0b"     # main text
INK_2   = "#52514e"     # secondary text (axes, notes)
GRID    = "#e4e3df"     # recessive gridlines
BLUE    = "#2a78d6"     # series colour 1
ORANGE  = "#eb6834"     # series colour 2
AQUA    = "#1baf7a"     # series colour 3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

con = duckdb.connect("data/mergemate.duckdb", read_only=True)

# ---- Chart 1: newcomer merge rate by year ----
df = con.sql("""
    SELECT YEAR(created_at) AS year, 100.0 * AVG(merged_30d) AS pct
    FROM newcomer_prs
    GROUP BY 1 ORDER BY 1
""").fetchdf()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(df["year"], df["pct"], color=BLUE, linewidth=2, marker="o", markersize=8)

# label only the first and last points
for i in (0, len(df) - 1):
    ax.annotate(f"{df['pct'][i]:.0f}%", (df["year"][i], df["pct"][i]),
                textcoords="offset points", xytext=(0, 10), ha="center", color=INK)

ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
ax.set_xticks(df["year"])
ax.set_title("First-time PRs merged within 30 days, by year", loc="left", color=INK, fontsize=13)
ax.text(0, -0.15, "Newcomer PRs across 50 open-source repos, 2021 to Oct 2026 (n = 11,442). *2026 = Jan to 10 Oct",
        transform=ax.transAxes, color=INK_2, fontsize=9)
ax.grid(axis="x", visible=False)
ax.set_xticklabels([str(y) if y != 2026 else "2026*" for y in df["year"]])
fig.tight_layout()
fig.savefig("docs/figures/01_merge_rate_by_year.png", dpi=150)
print("Saved docs/figures/01_merge_rate_by_year.png")

# ---- Chart 2: genuine return rate by first-PR outcome ----
df2 = con.sql("""
    WITH first_pr AS (
        SELECT repo, author_login, closed_at, state
        FROM newcomer_prs
        WHERE closed_at IS NOT NULL AND closed_at < '2026-08-12'
    ),
    second_pr AS (
        SELECT repo, author_login, created_at AS second_at
        FROM prs_flagged WHERE author_pr_seq = 2
    )
    SELECT CASE WHEN f.state = 'MERGED' THEN 'First PR merged' ELSE 'First PR rejected' END AS outcome,
           100.0 * AVG(CASE WHEN s.second_at >  f.closed_at + INTERVAL 1 DAY
                             AND s.second_at <= f.closed_at + INTERVAL 90 DAY
                            THEN 1 ELSE 0 END) AS pct_returned
    FROM first_pr f
    LEFT JOIN second_pr s USING (repo, author_login)
    WHERE s.second_at IS NULL OR s.second_at > f.closed_at      -- exclude parallel PRs
    GROUP BY 1 ORDER BY 1
""").fetchdf()

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(df2["outcome"], df2["pct_returned"], color=BLUE, width=0.45)

for bar, value in zip(bars, df2["pct_returned"]):                 # value on top of each bar
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%",
            ha="center", color=INK, fontsize=12)

ax.set_ylim(0, 30)
ax.set_yticks(range(0, 31, 10))
ax.set_yticklabels([f"{t}%" for t in range(0, 31, 10)])
ax.grid(axis="x", visible=False)
ax.set_title("Newcomers who come back with new work within 90 days", loc="left", color=INK, fontsize=13)
ax.text(0, -0.15, "Same-day resubmissions and PRs opened before the decision excluded. "
        "Association, not proven causation.", transform=ax.transAxes, color=INK_2, fontsize=9)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("docs/figures/02_return_rate_by_outcome.png", dpi=150)
print("Saved docs/figures/02_return_rate_by_outcome.png")
# ---- Chart 3: status at day 30, by year (stacked) ----
df3 = con.sql("""
    SELECT YEAR(created_at) AS year,
           100.0 * AVG(merged_30d) AS merged,
           100.0 * AVG(CASE WHEN state = 'CLOSED'
                             AND closed_at <= created_at + INTERVAL 30 DAY THEN 1 ELSE 0 END) AS rejected,
           100.0 * AVG(CASE WHEN closed_at IS NULL
                              OR closed_at > created_at + INTERVAL 30 DAY THEN 1 ELSE 0 END) AS undecided
    FROM newcomer_prs
    GROUP BY 1 ORDER BY 1
""").fetchdf()

labels = [str(y) if y != 2026 else "2026*" for y in df3["year"]]
fig, ax = plt.subplots(figsize=(8, 4.8))

bottom = [0] * len(df3)
for column, colour, name in [("merged", BLUE, "Merged"),
                             ("rejected", ORANGE, "Rejected"),
                             ("undecided", AQUA, "Undecided")]:
    ax.bar(labels, df3[column], bottom=bottom, color=colour, width=0.6,
           edgecolor=SURFACE, linewidth=2, label=name)        # 2px background-coloured gap
    bottom = [b + v for b, v in zip(bottom, df3[column])]

ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 25))
ax.set_yticklabels([f"{t}%" for t in range(0, 101, 25)])
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax.set_title("Status of first-time PRs 30 days after opening", loc="left", color=INK, fontsize=13)
ax.text(0, -0.24, "Same 30-day window for every year, so years are comparable. *2026 = Jan to 10 Oct",
        transform=ax.transAxes, color=INK_2, fontsize=9)

fig.tight_layout()
fig.savefig("docs/figures/03_status_at_30_days.png", dpi=150)
print("Saved docs/figures/03_status_at_30_days.png")
con.close()