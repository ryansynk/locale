from pathlib import Path

import altair as alt
import polars as pl
from jsonargparse import auto_cli


def main(results_dir: str, plots_dir: str = "plots"):
    results_dir: Path = Path(results_dir)
    plots_dir: Path = Path(plots_dir)
    if not plots_dir.is_dir():
        plots_dir.mkdir()
    data = []
    for f in list(results_dir.glob("*.parquet")):
        df = pl.read_parquet(f)
        data.append(df)
    data = pl.concat(data)
    for metric, title in [("hit_at_1", "Hit @ 1"), ("hit_at_5", "Hit @ 5")]:
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "identity_bin",
                    title="Sequence Identity",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(metric, title=title),
                xOffset="model",
                color="model",
            )
            .configure_axis(labelFontSize=15, titleFontSize=18)
            .configure_legend(labelFontSize=15, titleFontSize=18)
        )
        chart.save(plots_dir / f"{metric}.png")


if __name__ == "__main__":
    auto_cli(main)
