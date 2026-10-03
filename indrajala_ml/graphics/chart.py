# pyright: reportUnknownMemberType=false
# (matplotlib 3.8's annotations leave **kwargs untyped, so every Axes/Figure method reads as
# partially unknown; this module's own values are still checked where they're declared)
from collections.abc import Sequence

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend


def style_dark_legend(legend: Legend) -> None:
    """
    Styles a legend for the dark chart theme: a legend's frame and text stay light whatever the
    axes' facecolor.
    """

    legend.get_frame().set_facecolor("black")
    for text in legend.get_texts():
        text.set_color("white")


def plot_labeled_series(axes: Axes, results: Sequence[tuple[str, str, Sequence[float], Sequence[float]]]) -> None:
    """
    Plots each (label, color, x, y) series on axes, with a dark legend: the multi-series convergence
    chart of the sweep demos.
    """

    for label, color, x, y in results:
        axes.plot(x, y, color=color, label=label)

    style_dark_legend(axes.legend())


def disagreement_axis_bounds(x_max: float, log: bool = False) -> list[tuple[float, float]]:
    # a disagreement rate is always in [0, 1]; on a log-scaled axis 0.0 has no position, so
    # floor it just above zero instead
    return [(0.0, x_max), (1.0e-3, 1.0) if log else (0.0, 1.0)]


def new_convergence_chart_pair(linear_title: str, log_title: str, x_max: float) -> tuple[Axes, Axes]:
    """
    Linear and log-scaled disagreement axes, each on a new figure, for a convergence chart;
    callers plot on both.
    """

    linear_axes = new_axes(new_figure(linear_title), disagreement_axis_bounds(x_max), scaled=False)

    log_axes = new_axes(new_figure(log_title), disagreement_axis_bounds(x_max, log=True), scaled=False)
    log_axes.set_yscale("log")

    return linear_axes, log_axes


def place_tk_window(axes: Axes, geometry: str) -> None:
    """Moves axes' figure window to a Tk geometry string ("+x+y"); needs the TkAgg backend."""
    # imported here: only a TkAgg demo calls this, and the module imports tkinter
    from matplotlib.backends._backend_tk import FigureManagerTk

    figure = axes.figure
    assert figure is not None
    manager = figure.canvas.manager
    assert isinstance(manager, FigureManagerTk), f"place_tk_window needs the TkAgg backend; got {manager!r}"
    manager.window.wm_geometry(geometry)


def new_figure(label: str) -> Figure:
    figure = pyplot.figure(label)
    figure.patch.set_facecolor("xkcd:black")
    return figure


def new_axes(figure: Figure, bounds: list[tuple[float, float]] | None = None, scaled: bool = True) -> Axes:

    axes = figure.add_subplot(111)
    axes.set_facecolor("xkcd:black")

    axes.grid(True, which="both")

    if bounds:
        axes.set_xlim(bounds[0])
        axes.set_ylim(bounds[1])

    if scaled:
        axes.set_aspect("equal", adjustable="box")

    axes.spines["bottom"].set_color("white")
    axes.spines["top"].set_color("white")
    axes.spines["left"].set_color("white")
    axes.spines["right"].set_color("white")

    axes.xaxis.label.set_color("white")
    axes.yaxis.label.set_color("white")
    axes.tick_params(axis="x", colors="white")
    axes.tick_params(axis="y", colors="white")

    return axes
