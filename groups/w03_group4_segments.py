"""Group 4 (w03) - segments: sorts the gift shop's customers into k segments with k-means.

Authors: Zuzanna, Jan, Lena
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml_base import Analysis

# the colour of segment 0, 1, 2, ... in plot() (the first four are easy to tell apart, also for colour-blind readers)
COLORS = ["#2a78d6", "#eda100", "#e87ba4", "#008300", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]


class CustomerSegments(Analysis):
    """Puts every customer into one of k segments (k-means on recency_days, orders, revenue).

    Usage:
        model = CustomerSegments(customers, k=4, scale=True).fit()
        model.results_      # Series: the segment number (0 .. k-1) of every customer
        model.profile()     # DataFrame: one row per segment (size, averages, % who came back)
        model.plot()        # scatter orders vs revenue, coloured by segment

    Args:
        data (DataFrame): one row per customer, with the columns recency_days, orders, revenue
            (numbers >= 0, nothing missing). A column `returned` (0/1) is only needed for
            returned_pct in profile().
        k (int): how many segments, at least 2 and at most the number of customers. Default 4.
        scale (bool): True (default) = log1p + StandardScaler before k-means, so the three columns
            count equally. False = the raw numbers (then revenue decides almost everything).

    Raises:
        TypeError: if data is not a DataFrame
        ValueError: if a required column is missing, has missing or negative values,
            or if k is not a whole number between 2 and the number of customers
    """
    required_columns = ["recency_days", "orders", "revenue"]

    def __init__(self, data, k=4, scale=True):
        """Checks the table, k and scale and stores them (see the class docstring for the arguments)."""
        super().__init__(data)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            raise ValueError(f"k must be a whole number, got {k!r}")
        if k < 2:
            raise ValueError(f"k must be at least 2 (one segment = no segmentation), got {k}")
        if k > len(self.data):
            raise ValueError(f"k = {k} is more than the {len(self.data)} customers in the table")
        table = self.data[self.required_columns]
        if table.isna().any().any() or (table < 0).any().any():
            raise ValueError("recency_days, orders and revenue must be numbers >= 0 without missing values")
        self.k = int(k)
        self.scale = bool(scale)

    def fit(self):
        """Runs k-means on recency_days, orders and revenue.

        With scale=True the columns are first log-transformed (log1p tames the long tail) and
        standardised (mean 0, std 1). Stores the segment number of every customer in
        self.results_ (a Series, index = CustomerID), the fitted KMeans in self.model_,
        and returns self.
        """
        table = self.data[self.required_columns]
        if self.scale:
            X = StandardScaler().fit_transform(np.log1p(table))
        else:
            X = table.to_numpy(dtype=float)
        self.model_ = KMeans(n_clusters=self.k, n_init=10, random_state=0).fit(X)
        self.results_ = pd.Series(self.model_.labels_, index=self.data.index, name="segment")
        return self

    def profile(self):
        """Describes every segment in one row.

        Returns:
            DataFrame: index = segment number, columns customers (how many), recency_days, orders,
            revenue (the means of the segment) and returned_pct (% who came back; NaN if the table
            has no `returned` column). Sorted by revenue, the most valuable segment first.

        Raises:
            RuntimeError: if fit() was not called
        """
        self.check_fitted()
        groups = self.data.groupby(self.results_.to_numpy())
        table = groups[self.required_columns].mean()
        table.insert(0, "customers", groups.size())
        if "returned" in self.data.columns:
            table["returned_pct"] = groups["returned"].mean() * 100
        else:
            table["returned_pct"] = np.nan
        table.index.name = "segment"
        return table.sort_values("revenue", ascending=False).round(1)

    def plot(self):
        """Scatter plot of orders vs revenue (both axes on a log scale), coloured by segment.

        Returns:
            matplotlib Axes (the figure is ax.figure)

        Raises:
            RuntimeError: if fit() was not called
        """
        self.check_fitted()
        labels = self.results_.to_numpy()
        rng = np.random.default_rng(0)
        order = rng.permutation(len(labels))            # draw the customers in random order, so no segment hides another
        jitter = rng.uniform(0.88, 1.12, len(labels))   # orders are whole numbers: spread the points of one stripe a little
        colors = np.array(COLORS)[labels % len(COLORS)]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter((self.data["orders"] * jitter).iloc[order], self.data["revenue"].iloc[order],
                   c=colors[order], s=12, alpha=0.6, linewidths=0)
        for segment in range(self.k):                   # empty scatters: only for the legend
            ax.scatter([], [], color=COLORS[segment % len(COLORS)],
                       label=f"{segment}  ({(labels == segment).sum():,} customers)")
        scaled = "log1p + standardised" if self.scale else "raw numbers"
        ax.set(xscale="log", yscale="log", xlabel="orders (log scale, slightly jittered)",
               ylabel="revenue, GBP (log scale)", title=f"{self.k} customer segments, k-means on {scaled}")
        ax.legend(title="segment", frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        return ax
