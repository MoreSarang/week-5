import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure


def survival_demographics(file_path: str) -> pd.DataFrame:
    """
    Analyze Titanic survival patterns by class, sex, and age group.

    Args:
        file_path (str): Path to the Titanic CSV dataset.

    Returns:
        pd.DataFrame: Summary table with passenger counts, survivors,
                      and survival rates by demographic groups.
    """
    df = pd.read_csv(file_path)

    # Ensure Sex values are lowercased and stripped
    df['Sex'] = df['Sex'].str.lower().str.strip()

    # Create age categories (handle missing Age as NaN)
    age_bins = [0, 12, 19, 59, float("inf")]
    age_labels = ["Child", "Teen", "Adult", "Senior"]
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    # Create a MultiIndex of all possible class/sex/age_group combinations
    pclass = sorted(df['Pclass'].unique())
    sex = sorted(df['Sex'].unique())
    age_group = age_labels

    idx = pd.MultiIndex.from_product(
        [pclass, sex, age_group],
        names=['Pclass', 'Sex', 'age_group']
    )

    grouped = (
        df.groupby(['Pclass', 'Sex', 'age_group'])
        .agg(n_passengers=('PassengerId', 'count'),
             n_survivors=('Survived', 'sum'))
        .reindex(idx, fill_value=0)
        .reset_index()
    )

    grouped['survival_rate'] = grouped.apply(
        lambda row: round(row['n_survivors'] / row['n_passengers'], 2) if row['n_passengers'] > 0 else 0,
        axis=1
    )

    grouped = grouped.sort_values(by=["Pclass", "Sex", "age_group"]).reset_index(drop=True)
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
    fig.update_layout(
        yaxis=dict(title="Survival Rate", range=[0, 1]),
        bargap=0.2,
    )

    return fig


def family_groups(file_path: str) -> pd.DataFrame:
    """
    Analyze the relationship between family size, passenger class, and ticket fare.

    Args:
        file_path (str): Path to Titanic CSV.

    Returns:
        pd.DataFrame: Grouped stats by family size and passenger class.
    """
    df = pd.read_csv(file_path)
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


def last_names(file_path: str) -> pd.Series:
    """
    Extract and count last names from Titanic dataset.

    Args:
        file_path (str): Path to Titanic CSV.

    Returns:
        pd.Series: Last name counts (sorted by count descending).
    """
    df = pd.read_csv(file_path)
    df["last_name"] = df["Name"].str.split(",").str[0].str.strip()
    counts = df["last_name"].value_counts()
    return counts
