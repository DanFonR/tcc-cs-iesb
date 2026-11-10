#!/usr/bin/env python3
"""Analise dos resultados do TCC.

Compara equações normais (NE, GCC/Clang em O0, O1 e O3+native) contra a
decomposição QR de Householder (LAPACKE/OpenBLAS), usada como referência.

Uso, a partir da raiz do projeto::

    python3 analysis.py

Sem argumentos, procura o results.csv e o stability.csv mais recentes em
``data/results/`` (ou no diretorio atual). Opcoes: ``--results``,
``--stability``, ``--outdir`` e ``--dtype``.

Estrutura esperada dos CSVs (colunas usadas):
    lr_method        {NE, QR}
    compiler         {gcc, clang}
    optim            {O0, O1, native}
    data_type        {well, ill} em results.csv;
    {stab_e0, stab_e3, ...} em stability.csv
    n_samples, n_features, condition_num_2
    total_time       tempo total em segundos (soma das fases)
    formation_time, decomposition_time, solve_time   tempos por fase (s)
    relative_coef_err_l2, coef_rmse, coef_mae, max_abs_coef_err,
    relative_resid_err_l2, resid_rmse, resid_mae, max_abs_resid_err
        colunas de erro
        (results.csv: contra o beta verdadeiro;
        stability.csv: contra a referencia long double)

Os tempos sao interpretados como segundos.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter

type linestyle = tuple[int, tuple[int, float]] | str

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
TIME_COL: str = "total_time"
ERR_COL: str = "relative_coef_err_l2"
KAPPA_COL: str = "condition_num_2"
UNIT_ROUNDOFF: float = 2.0 ** -53  # u, para double

PHASE_COLS: tuple[str, ...] = (
    "formation_time", "decomposition_time", "solve_time"
)
PHASE_LABELS: dict[str, str] = {
    "formation_time": "Formação (XtX, Xtb)",
    "decomposition_time": "Decomposição (Cholesky / dgeqrf)",
    "solve_time": "Solução (substituições / dormqr+dtrtrs)",
}
# Colunas de erro comparadas entre NE e QR
# (min_abs_* ficam de fora: nao medem erro util;
# coef_rmse = coef_err_l2 / sqrt(p), portanto sao redundantes entre si).
ERROR_COLS: tuple[str, ...] = (
    "relative_coef_err_l2", "coef_rmse", "coef_mae", "max_abs_coef_err",
    "relative_resid_err_l2", "resid_rmse", "resid_mae", "max_abs_resid_err",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "lr_method", "compiler", "optim", "data_type",
    "n_samples", "n_features", "condition_num_2", "total_time",
)

OPTIM_MAP: dict[str, str] = {"o0": "O0", "o1": "O1", "native": "O3+native"}
OPTIM_ORDER: list[str] = ["O0", "O1", "O3+native"]
COMPILER_LABEL: dict[str, str] = {
    "gcc": "GCC",
    "clang": "Clang",
    "openblas": "OpenBLAS"
}
QR_LABEL: str = "OpenBLAS (QR)"

COLORS: dict[str, str] = {
    "gcc": "#1f77b4",
    "clang": "#ff7f0e",
    "openblas": "#000000"
}
LINESTYLES: dict[str, linestyle] = {
    "O0": (0, (1, 1.5)),
    "O1": (0, (4, 2)),
    "O3+native": "-"
}
MARKERS: dict[str, str] = {"O0": "o", "O1": "s", "O3+native": "^"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


# ----------------------------------------------------------------------------
# Utilitarios
# ----------------------------------------------------------------------------
def set_log_ticks(ax: Axes, ticks: Sequence[float]) -> None:
    """Ticks decimais fixos num eixo y log, sem alterar os limites."""
    limits = ax.get_ylim()
    ax.set_yticks(list(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(limits)


def fmt_n(value: float) -> str:
    """Formata n com separador de milhar em estilo brasileiro (1.000.000)."""
    return f"{int(value):,}".replace(",", ".")


def find_latest(
    patterns: Sequence[str],
    exclude: Sequence[str] = ("run1",)
) -> Path | None:
    """
    Retorna o arquivo mais recente (ordem alfabética do nome)
    que casa com algum padrão.
    """
    files: list[str] = []

    for pattern in patterns:
        files.extend(glob.glob(pattern))

    candidates = [
        Path(f) for f in files if not any(x in Path(f).name for x in exclude)
    ]

    return (
        sorted(candidates, key=lambda p: p.name)[-1]
        if candidates else None
    )


def save_figure(fig: Figure, outdir: Path, name: str) -> None:
    """Salva a figura em PNG e a fecha."""
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {name}")


# ----------------------------------------------------------------------------
# Carga e normalizacao
# ----------------------------------------------------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV de resultados e adiciona colunas derivadas.

    Colunas adicionadas:
        family       'NE' ou 'QR'
        comp         'gcc', 'clang' ou 'openblas' (QR)
        opt          'O0', 'O1', 'O3+native' ou '-' (QR)
        variant      rotulo legivel, ex.: 'GCC O1'
        time_s       tempo total em segundos
    """
    df = pd.read_csv(path)

    missing = [cols for cols in REQUIRED_COLUMNS if cols not in df.columns]

    if missing:
        raise SystemExit(f"{path}: colunas ausentes: {missing}")

    for col in ("lr_method", "compiler", "optim", "data_type"):
        df[col] = df[col].astype(str).str.strip()

    methods = set(df["lr_method"].str.upper())

    if not methods <= {"NE", "QR"}:
        raise SystemExit(
            f"{path}: valores inesperados em lr_method: {sorted(methods)}"
        )

    is_qr = df["lr_method"].str.upper() == "QR"
    unknown_optim = set(df.loc[~is_qr, "optim"].str.lower()) - set(OPTIM_MAP)

    if unknown_optim:
        raise SystemExit(
            f"{path}: valores inesperados em optim: {sorted(unknown_optim)}"
        )

    df["family"] = np.where(is_qr, "QR", "NE")
    df["comp"] = np.where(is_qr, "openblas", df["compiler"].str.lower())
    df["opt"] = np.where(is_qr, "-", df["optim"].str.lower().map(OPTIM_MAP))
    df["variant"] = np.where(
        is_qr,
        QR_LABEL,
        df["comp"].map(COMPILER_LABEL) + " " + df["opt"],
    )
    df["time_s"] = df[TIME_COL].astype(float)

    return df


# ----------------------------------------------------------------------------
# Estatistica descritiva
# ----------------------------------------------------------------------------
def describe_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estatísticas por grupo (variante, tipo de dado, n, p) sobre as execuções.
    """
    keys = [
        "family", "variant", "comp", "opt",
        "data_type", "n_samples", "n_features"
    ]
    grouped = df.groupby(keys)["time_s"]
    agg = grouped.agg(
        n_runs="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    ).reset_index()
    agg["iqr"] = agg["q3"] - agg["q1"]
    agg["cv_pct"] = 100.0 * agg["std"] / agg["mean"]

    work = agg["n_samples"] * agg["n_features"] ** 2  # custo O(n p^2) do NE

    for col in ("median", "q1", "q3"):
        agg[f"{col}_norm"] = agg[col] * 1e9 / work  # ns por n*p^2

    return agg


def compute_speedups(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Speedup do NE entre niveis de otimização (razão das medianas).
    """
    ne = agg[agg["family"] == "NE"]
    wide = ne.pivot_table(
        index=["comp", "data_type", "n_samples", "n_features"],
        columns="opt",
        values="median",
    ).reset_index()

    parts: list[pd.DataFrame] = []
    transitions = {
        "O0 → O1": ("O0", "O1"),
        "O1 → O3+native": ("O1", "O3+native")
    }

    for label, (slow, fast) in transitions.items():
        part = wide[["comp", "data_type", "n_samples", "n_features"]].copy()
        part["transition"] = label
        part["speedup"] = wide[slow] / wide[fast]

        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def compute_gaps(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Gap NE/QR: mediana do NE dividida pela mediana do QR na mesma célula.
    """
    cols = [
        "variant", "comp", "opt", "data_type",
        "n_samples", "n_features", "median"
    ]
    ne = agg.loc[
        agg["family"] == "NE",
        cols,
    ]
    qr = agg.loc[
        agg["family"] == "QR",
        ["data_type", "n_samples", "n_features", "median"],
    ].rename(columns={"median": "qr_median"})
    merged = ne.merge(
        qr, on=["data_type", "n_samples", "n_features"], how="inner"
    )
    merged["gap"] = merged["median"] / merged["qr_median"]

    return merged


def print_summaries(
    agg: pd.DataFrame, speedups: pd.DataFrame,
    gaps: pd.DataFrame, outdir: Path
) -> None:
    """
    Imprime e salva os resumos (medianas sobre p) usados no texto do TCC.
    """
    grouped = speedups.groupby([
        "data_type", "comp", "transition", "n_samples"
    ])["speedup"]
    sp_tab = (
        grouped
        .median()
        .unstack("n_samples")
    )
    gap_tab = (
        gaps.groupby(["data_type", "variant", "n_samples"])["gap"]
        .median()
        .unstack("n_samples")
    )
    cv_tab = (
        agg.groupby(["data_type", "n_samples"])["cv_pct"]
        .agg(["median", "max"]).round(2)
    )

    print("\nSPEEDUP do NE (mediana sobre p):\n", sp_tab.round(2))
    print("\nGAP NE/QR (mediana sobre p):\n", gap_tab.round(2))
    print("\nCV (%) das execuções, por n:\n", cv_tab)

    medians = agg.pivot_table(
        index=["variant", "n_samples", "n_features"],
        columns="data_type", values="median"
    )
    if {"ill", "well"} <= set(medians.columns):
        ratio = (
            (medians["ill"] / medians["well"])
            .groupby("variant").agg(["min", "max"]).round(3)
        )

        print("\nRazão de tempo ill/well por variante:\n", ratio)

    sp_tab.to_csv(outdir / "resumo_speedup.csv")
    gap_tab.to_csv(outdir / "resumo_gap.csv")


def describe_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mediana de cada fase (formação, decomposição, solução)
    e sua fração do total.
    """
    keys = [
        "family", "variant", "comp", "opt",
        "data_type", "n_samples", "n_features"
    ]
    phases = df.groupby(keys)[list(PHASE_COLS)].median().reset_index()
    phases["total"] = phases[list(PHASE_COLS)].sum(axis=1)

    for col in PHASE_COLS:
        phases[f"{col}_frac"] = phases[col] / phases["total"]

    return phases


def compare_error_columns(df: pd.DataFrame, outdir: Path) -> None:
    """
    Compara as colunas de erro do results.csv entre NE e QR
    (razão NE/QR por célula).

    No results.csv os erros são contra o beta verdadeiro (dominados pelo ruído
    estatístico) e os resíduos são normas do resíduo;
    razão ~1 significa que os dois métodos produzem a mesma solução nesses
    conjuntos bem condicionados.
    """
    cols = [c for c in ERROR_COLS if c in df.columns]
    keys = ["data_type", "n_samples", "n_features"]
    per_variant = (
        df.groupby(["family", "variant"] + keys)[cols]
        .median().reset_index()
    )
    ne = (
        per_variant[per_variant["family"] == "NE"]
        .groupby(keys)[cols].median()
    )
    qr = (
        per_variant[per_variant["family"] == "QR"]
        .groupby(keys)[cols].median()
    )
    ratio = (ne / qr).reset_index()

    ratio.to_csv(outdir / "tabela_equivalencia_erros.csv", index=False)

    summary = pd.DataFrame({
        "min": ratio[cols].min(),
        "mediana": ratio[cols].median(),
        "max": ratio[cols].max(),
        "max|razão-1|": (ratio[cols] - 1.0).abs().max(),
    })

    print("\nRazão NE/QR das colunas de erro (results.csv):\n", summary)


# ----------------------------------------------------------------------------
# Figuras de tempo
# ----------------------------------------------------------------------------
def line_style(comp: str, opt: str) -> dict[str, str | linestyle | float]:
    """
    Estilo de linha: cor por compilador,
    tracejado por otimizacao, QR em destaque.
    """
    if comp == "openblas":
        return {
            "color": COLORS["openblas"],
            "ls": "-", "lw": 2.4,
            "marker": "D", "ms": 4, "zorder": 5
        }
    return {
        "color": COLORS[comp], "ls": LINESTYLES[opt], "lw": 1.5,
        "marker": MARKERS[opt], "ms": 3.5, "zorder": 3,
    }


def variant_order(agg: pd.DataFrame) -> list[str]:
    """Ordem de legenda: GCC (O0, O1, native), Clang (idem), OpenBLAS."""
    info = agg[["variant", "comp", "opt"]].drop_duplicates().copy()
    info["c"] = info["comp"].map({"gcc": 0, "clang": 1, "openblas": 2})
    info["o"] = (
        info["opt"].map({k: i for i, k in enumerate(OPTIM_ORDER)}).fillna(9)
    )

    return list(info.sort_values(["c", "o"])["variant"])


def plot_lines(
    ax: Axes, sub: pd.DataFrame,
    xcol: str, ycol: str = "median"
) -> None:
    """Desenha uma linha por variante, com faixa interquartil sombreada."""
    lo, hi = ("q1", "q3") if ycol == "median" else ("q1_norm", "q3_norm")

    for variant, g in sub.groupby("variant"):
        g = g.sort_values(xcol)
        first = g.iloc[0]
        style = line_style(first["comp"], first["opt"])

        ax.plot(g[xcol], g[ycol], label=variant, **style)
        ax.fill_between(
            g[xcol], g[lo], g[hi],
            color=style["color"], alpha=0.12, lw=0
        )


def add_shared_legend(
    fig: Figure, ax: Axes,
    order: Sequence[str], y_anchor: float
) -> None:
    """Legenda única na base da figura, na ordem desejada."""
    handles, labels = ax.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    names = [v for v in order if v in lookup]

    fig.legend(
        [lookup[n] for n in names], names,
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, y_anchor),
    )


def fig_time_vs_n(agg: pd.DataFrame, dtype: str, outdir: Path) -> None:
    """Tempo x n (log-log) para o menor e o maior p."""
    data = agg[agg["data_type"] == dtype]
    ps = sorted(data["n_features"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, p in zip(axes, (ps[0], ps[-1])):
        plot_lines(ax, data[data["n_features"] == p], "n_samples")
        ax.set(
            xscale="log", yscale="log",
            xlabel="Amostras (n)", title=f"p = {p}"
        )

    axes[0].set_ylabel("Tempo mediano (s)")
    add_shared_legend(fig, axes[0], variant_order(data), y_anchor=-0.2)
    save_figure(fig, outdir, "fig1_tempo_vs_n")


def fig_time_vs_p(
    agg: pd.DataFrame, dtype: str, outdir: Path,
    ycol: str = "median", name: str = "fig2_tempo_vs_p",
    ylabel: str = "Tempo mediano (s)",
) -> None:
    """Tempo (ou tempo normalizado por n*p^2) x p, um painel por n."""
    data = agg[agg["data_type"] == dtype]
    ns = sorted(data["n_samples"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    for ax, n in zip(axes.ravel(), ns):
        plot_lines(ax, data[data["n_samples"] == n], "n_features", ycol=ycol)
        ax.set(yscale="log", title=f"n = {fmt_n(n)}")

        if ycol == "median_norm":
            set_log_ticks(ax, [0.1, 0.2, 0.5, 1, 2])

    for ax in axes[-1]:
        ax.set_xlabel("Variáveis (p)")

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    add_shared_legend(fig, axes[0, 0], variant_order(data), y_anchor=0.0)
    save_figure(fig, outdir, name)


# ----------------------------------------------------------------------------
# Speedup e gap
# ----------------------------------------------------------------------------
def fig_speedup_bars(
    speedups: pd.DataFrame, dtype: str,
    outdir: Path
) -> None:
    """Barras de speedup por n e compilador (mediana sobre p, barra = IQR)."""
    data = speedups[speedups["data_type"] == dtype].copy()
    data["n"] = data["n_samples"].map(fmt_n)
    data["Compilador"] = data["comp"].map(COMPILER_LABEL)
    grid = sns.catplot(
        data=data, x="n", y="speedup", hue="Compilador",
        col="transition", kind="bar",
        estimator=np.median, errorbar=("pi", 50),
        palette={"GCC": COLORS["gcc"], "Clang": COLORS["clang"]},
        height=3.8, aspect=1.1, capsize=0.08,
    )

    for ax in grid.axes.ravel():
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Amostras (n)")

    grid.axes[0, 0].set_ylabel(
        "Speedup (mediana sobre p; barra = IQR entre p)"
    )
    grid.set_titles("{col_name}")
    save_figure(grid.figure, outdir, "fig3_speedup")


def draw_heatmap(
    ax: Axes, matrix: pd.DataFrame, color_values: np.ndarray, annot_fmt: str,
    cmap: str, vmin: float, vmax: float, show_ylabels: bool, show_xlabel: bool,
) -> None:
    """Mapa de calor n x p com valores anotados."""
    annot = np.array([
        [annot_fmt.format(v) for v in row] for row in matrix.to_numpy()
    ])

    sns.heatmap(
        pd.DataFrame(color_values, index=matrix.index, columns=matrix.columns),
        ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, annot=annot, fmt="",
        annot_kws={"fontsize": 6}, cbar=False, linewidths=0.3,
    )
    ax.set_xlabel("p" if show_xlabel else "")
    ax.tick_params(axis="x", labelsize=7)

    if show_ylabels:
        ax.set_ylabel("n")
        ax.set_yticklabels(
            [fmt_n(float(t.get_text())) for t in ax.get_yticklabels()],
            rotation=0
        )
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def pivot_np(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """
    Matriz com n nas linhas (crescente) e p nas colunas.
    """
    return (
        df.pivot_table(index="n_samples", columns="n_features", values=value)
        .sort_index()
    )


def fig_gap_heatmap(gaps: pd.DataFrame, dtype: str, outdir: Path) -> None:
    """Gap NE/QR: linhas = compilador, colunas = otimização."""
    data = gaps[gaps["data_type"] == dtype]
    vmin = min(0.0, np.log10(data["gap"].min()))  # 0 = paridade (gap = 1)
    vmax = np.log10(data["gap"].max())
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 5.5), gridspec_kw={"hspace": 0.3, "wspace": 0.08}
    )

    for i, comp in enumerate(("gcc", "clang")):
        for j, opt in enumerate(OPTIM_ORDER):
            cell = data[(data["comp"] == comp) & (data["opt"] == opt)]
            matrix = pivot_np(cell, "gap")

            draw_heatmap(
                axes[i, j], matrix, np.log10(matrix.to_numpy()),
                "{:.1f}", "mako_r", vmin, vmax,
                show_ylabels=(j == 0), show_xlabel=(i == 1),
            )
            axes[i, j].set_title(
                f"{COMPILER_LABEL[comp]} / {opt}", fontsize=10
            )

    mappable = plt.cm.ScalarMappable(
        cmap="mako_r", norm=plt.Normalize(vmin, vmax)
    )
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.8, pad=0.02)

    ticks = [t for t in (1, 2, 4, 8, 16) if vmin <= np.log10(t) <= vmax]
    cbar.set_ticks([np.log10(t) for t in ticks])
    cbar.set_ticklabels([str(t) for t in ticks])
    cbar.set_label("Gap NE/QR (escala log)")
    save_figure(fig, outdir, "fig4_gap_heatmap")


def fig_speedup_heatmap(
    speedups: pd.DataFrame, dtype: str, outdir: Path
) -> None:
    """Speedup O1 -> O3+native do NE, n x p, por compilador."""
    data = speedups[
        (speedups["data_type"] == dtype) &
        (speedups["transition"] == "O1 → O3+native")
    ]
    limit = float(
        np.log2(max(data["speedup"].max(), 1.0 / data["speedup"].min()))
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.4), gridspec_kw={"wspace": 0.08}
    )

    for j, comp in enumerate(("gcc", "clang")):
        matrix = pivot_np(data[data["comp"] == comp], "speedup")

        draw_heatmap(
            axes[j], matrix, np.log2(matrix.to_numpy()), "{:.2f}", "RdBu",
            -limit, limit, show_ylabels=(j == 0), show_xlabel=True,
        )
        axes[j].set_title(
            f"{COMPILER_LABEL[comp]}: O1 → O3+native", fontsize=10
        )

    mappable = plt.cm.ScalarMappable(
        cmap="RdBu", norm=plt.Normalize(-limit, limit)
    )
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.85, pad=0.02)

    cbar.set_label("log2(speedup)  (> 0: native mais rapido)")
    save_figure(fig, outdir, "fig5_speedup_o1_native_heatmap")


# ----------------------------------------------------------------------------
# Dispersao das execucoes
# ----------------------------------------------------------------------------
def fig_boxplot_dispersion(df: pd.DataFrame, dtype: str, outdir: Path) -> None:
    """
    Boxplot do tempo de cada execução relativo à mediana do próprio grupo.
    """
    data = df[df["data_type"] == dtype].copy()
    data["rel"] = (
        data["time_s"] /
        (
            data.groupby(["variant", "n_samples", "n_features"])["time_s"]
            .transform("median")
        )
    )
    data["Grupo"] = np.where(data["family"] == "QR", QR_LABEL, data["opt"])
    data["n"] = data["n_samples"].map(fmt_n)

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(
        data=data, x="n", y="rel", hue="Grupo",
        hue_order=OPTIM_ORDER + [QR_LABEL], ax=ax, fliersize=1.5,
        linewidth=0.8, medianprops={"color": "k", "linewidth": 1.3},
        palette=["#9ecae1", "#fdae6b", "#a1d99b", "#bdbdbd"],
    )
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set(
        yscale="log", xlabel="Amostras (n)",
        ylabel="Tempo / mediana do próprio grupo"
    )
    set_log_ticks(ax, [0.5, 0.75, 1, 1.5, 2, 2.5])
    ax.legend(
        title="", ncol=4, frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, -0.2)
    )
    save_figure(fig, outdir, "fig6_boxplot_dispersao")


def fig_phase_breakdown(
    phases: pd.DataFrame, dtype: str, outdir: Path
) -> None:
    """Fração do tempo em cada fase, para o maior n e o menor/maior p."""
    data = phases[phases["data_type"] == dtype]
    n = int(data["n_samples"].max())
    ps = sorted(data["n_features"].unique())
    order = variant_order(data)
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)

    for ax, p in zip(axes, (ps[0], ps[-1])):
        sub = (
            data[
                (data["n_samples"] == n) & (data["n_features"] == p)
            ].set_index("variant").reindex(order)
        )
        left = np.zeros(len(order))

        for col, color in zip(PHASE_COLS, colors):
            frac = 100.0 * sub[f"{col}_frac"].to_numpy()

            ax.barh(
                order, frac, left=left,
                color=color, label=PHASE_LABELS[col]
            )

            left += frac

        for y, total in enumerate(sub["total"]):
            ax.text(101.5, y, f"{total:#.3g} s", va="center", fontsize=8)

        ax.set(
            xlim=(0, 122), xlabel="% do tempo total",
            title=f"n = {fmt_n(n)}, p = {p}"
        )
        ax.set_xticks(range(0, 101, 25))

    axes[0].invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, -0.08)
    )
    save_figure(fig, outdir, "fig8_fases")


# ----------------------------------------------------------------------------
# Estabilidade numérica
# ----------------------------------------------------------------------------
def fig_stability(stab: pd.DataFrame, outdir: Path) -> None:
    """
    Erro relativo dos coeficientes x kappa_2, com retas kappa*u e kappa^2*u.
    """
    grouped = (
        stab.groupby(["family", "n_features", "data_type"])
        .agg(
            kappa=(KAPPA_COL, "median"),
            err=(ERR_COL, "median"),
            nan_frac=(ERR_COL, lambda x: float(x.isna().mean())),
            coef_rmse=("coef_rmse", "median"),
            coef_mae=("coef_mae", "median"),
            max_abs_coef_err=("max_abs_coef_err", "median"),
            relative_resid_err_l2=("relative_resid_err_l2", "median"),
        )
        .reset_index()
    )
    grouped["failed"] = grouped["nan_frac"] >= 0.5
    grouped["err_over_ku"] = (
        grouped["err"] / (grouped["kappa"] * UNIT_ROUNDOFF)
    )
    grouped["err_over_k2u"] = (
        grouped["err"] / (grouped["kappa"] ** 2 * UNIT_ROUNDOFF)
    )
    grouped.to_csv(outdir / "tabela_estabilidade.csv", index=False)
    print("  tabela: tabela_estabilidade.csv")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    kappas = np.logspace(
        np.log10(grouped["kappa"].min()) - 0.3,
        np.log10(grouped["kappa"].max()) + 0.3,
        50
    )
    ax.plot(
        kappas, UNIT_ROUNDOFF * kappas,
        color="gray", ls=":", lw=1.2, label=r"$\kappa\,u$"
    )
    ax.plot(
        kappas, UNIT_ROUNDOFF * kappas ** 2,
        color="gray", ls="--", lw=1.2, label=r"$\kappa^2\,u$"
    )

    markers = dict(
        zip(
            sorted(grouped["n_features"].unique()),
            ["o", "s", "^", "v", "D"]
        )
    )
    families = (
        ("NE", COLORS["clang"], "Eq. normais"),
        ("QR", COLORS["openblas"], "QR (OpenBLAS)")
    )

    for family, color, name in families:
        for p, marker in markers.items():
            sub = grouped[
                (grouped["family"] == family) &
                (grouped["n_features"] == p) &
                ~grouped["failed"]
            ].sort_values("kappa")

            ax.plot(
                sub["kappa"], sub["err"],
                marker=marker, color=color, ms=5,
                lw=0.8, alpha=0.9, label=f"{name}, p={p}"
            )

    failed = grouped[(grouped["family"] == "NE") & grouped["failed"]]

    if not failed.empty:
        ax.scatter(
            failed["kappa"], np.full(len(failed), 0.5),
            marker="X", color="red", s=60, zorder=6, label="NE: falha (NaN)"
        )

    ax.set(
        xscale="log", yscale="log",
        xlabel=r"Número de condição $\kappa_2(X)$",
        ylabel="Erro relativo dos coeficientes\n(vs. referência em long double)",
        ylim=(1e-17, 2)
    )
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    save_figure(fig, outdir, "fig7_estabilidade")

    ne = stab[(stab["family"] == "NE")].dropna(subset=[ERR_COL])
    spread = ne.groupby(["n_features", "data_type"])[ERR_COL].agg(
        lambda x: 100.0 * (x.max() - x.min()) / x.median()
    )

    print(
        f"  NE: variação máxima do erro entre variantes = {spread.max():.2f}%"
    )

    resid = grouped.pivot_table(
        index=["n_features", "data_type"],
        columns="family", values="relative_resid_err_l2"
    ).dropna()
    max_dev = float(((resid["NE"] / resid["QR"]) - 1.0).abs().max())

    print(
        "  resíduo relativo NE vs QR (células sem NaN): " +
        f"max|razão-1| = {max_dev:.2e}"
    )


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    parser.add_argument(
        "--results", type=Path,
        default=None, help="CSV de tempos"
    )
    parser.add_argument(
        "--stability", type=Path,
        default=None, help="CSV de estabilidade"
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("analysis/figs"), help="diretório de saída"
    )
    parser.add_argument(
        "--dtype", choices=("well", "ill"),
        default="well", help="tipo de dado usado nas figuras de tempo"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    results_path = (
        args.results or
        find_latest(["data/results/*results.csv", "*results.csv"])
    )
    stability_path = (
        args.stability or
        find_latest(["data/results/*stability*.csv", "*stability*.csv"])
    )

    if results_path is None:
        raise SystemExit("results.csv nao encontrado; use --results CAMINHO")

    print(f"results:   {results_path}\nstability: {stability_path}")

    df = load_csv(results_path)
    print(f"  linhas: {len(df)}; variantes: {sorted(df['variant'].unique())}")
    print(f"  NaN em {TIME_COL}: {int(df['time_s'].isna().sum())}")

    agg = describe_times(df)
    agg.to_csv(args.outdir / "tabela_descritiva.csv", index=False)
    speedups = compute_speedups(agg)
    gaps = compute_gaps(agg)
    speedups.to_csv(args.outdir / "tabela_speedup.csv", index=False)
    gaps.to_csv(args.outdir / "tabela_gap.csv", index=False)
    print_summaries(agg, speedups, gaps, args.outdir)

    phases = describe_phases(df)
    phases.to_csv(args.outdir / "tabela_fases.csv", index=False)
    compare_error_columns(df, args.outdir)

    print("\nFiguras:")
    fig_time_vs_n(agg, args.dtype, args.outdir)
    fig_time_vs_p(agg, args.dtype, args.outdir)
    fig_time_vs_p(
        agg, args.dtype, args.outdir, ycol="median_norm",
        name="fig2b_ns_por_np2", ylabel=r"Tempo mediano por $n\,p^2$ (ns)"
    )
    fig_speedup_bars(speedups, args.dtype, args.outdir)
    fig_gap_heatmap(gaps, args.dtype, args.outdir)
    fig_speedup_heatmap(speedups, args.dtype, args.outdir)
    fig_boxplot_dispersion(df, args.dtype, args.outdir)
    fig_phase_breakdown(phases, args.dtype, args.outdir)

    if stability_path is not None and stability_path.exists():
        fig_stability(load_csv(stability_path), args.outdir)
    else:
        print("  (stability.csv não encontrado; figura 7 omitida)")


if __name__ == "__main__":
    main()
