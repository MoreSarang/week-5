import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure


def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze Titanic survival patterns by class, sex, and age group.

    Args:
        df (pd.DataFrame): Titanic dataset as DataFrame.

    Returns:
        pd.DataFrame: Summary table with passenger counts, survivors,
                      and survival rates by demographic groups.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    age_labels = ["Child", "Teen", "Adult", "Senior"]
    pclass_categories = [1, 2, 3]
    sex_categories = ["male", "female"]

    # Create categorical age_group column with categories defined explicitly
    df["age_group"] = pd.cut(
        df["Age"], bins=[0, 12, 19, 59, float("inf")], labels=age_labels
    )
    df["age_group"] = pd.Categorical(
        df["age_group"], categories=age_labels, ordered=True
    )

    # Normalize and categorize Pclass and Sex columns
    df["Pclass"] = pd.Categorical(
        df["Pclass"], categories=pclass_categories, ordered=True
    )
    df["Sex"] = df["Sex"].str.lower().str.strip()
    df["Sex"] = pd.Categorical(df["Sex"], categories=sex_categories)

    # Create a full MultiIndex for all possible groups
    idx = pd.MultiIndex.from_product(
        [pclass_categories, sex_categories, age_labels],
        names=["Pclass", "Sex", "age_group"]
    )

    # Group by and aggregate survival info, reindex to include all groups
    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"], observed=False)
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum")
        )
        .reindex(idx, fill_value=0)
        .reset_index()
    )

    # Calculate survival rate safely
    grouped["survival_rate"] = grouped.apply(
        lambda row: round(row["n_survivors"] / row["n_passengers"], 2)
        if row["n_passengers"] > 0
        else 0,
        axis=1,
    )

    return grouped


def visualize_demographic(summary_df: pd.DataFrame) -> Figure:
    """
    Create a grouped bar chart of Titanic survival demographics.

    Args:
        summary_df (pd.DataFrame): DataFrame returned by survival_demographics().

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = px.bar(
        summary_df,
        x="age_group",
        y="survival_rate",
        color="Sex",
        barmode="group",
        facet_col="Pclass",
        text="survival_rate",
        category_orders={
            "age_group": ["Child", "Teen", "Adult", "Senior"],
            "Sex": ["male", "female"],
            "Pclass": [1, 2, 3],
        },
        title="Titanic Survival Rate by Class, Sex, and Age Group",
        labels={
            "age_group": "Age Group",
            "survival_rate": "Survival Rate",
            "Sex": "Gender",
            "Pclass": "Passenger Class",
        },
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis=dict(title="Survival Rate", range=[0, 1]), bargap=0.2)

    return fig


def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze relationship between family size, passenger class, and ticket fare.

    Args:
        df (pd.DataFrame): Titanic dataset as DataFrame.

    Returns:
        pd.DataFrame: Aggregated data grouped by family size and passenger class.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    grouped = (
        df.groupby(["family_size", "Pclass"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
        .sort_values(by=["Pclass", "family_size"])
    )

    return grouped


def last_names(df: pd.DataFrame) -> pd.Series:
    """
    Extract last names from the Titanic dataset and count their occurrences.

    Args:
        df (pd.DataFrame): Titanic dataset as DataFrame.

    Returns:
        pd.Series: Last name counts indexed by last name.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    df["last_name"] = df["Name"].str.split(",").str[0].str.strip()
    counts = df["last_name"].value_counts()
    return counts
