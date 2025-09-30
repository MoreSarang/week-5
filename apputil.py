import pandas as pd
import plotly.express as px


def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_labels = ["Child", "Teen", "Adult", "Senior"]
    pclass_categories = [1, 2, 3]
    sex_categories = ["male", "female"]

    df["age_group"] = pd.cut(
        df["Age"], bins=[0, 12, 19, 59, float("inf")], labels=age_labels
    )
    df["age_group"] = pd.Categorical(
        df["age_group"], categories=age_labels, ordered=True
    )

    df["Pclass"] = pd.Categorical(
        df["Pclass"], categories=pclass_categories, ordered=True
    )
    df["Sex"] = df["Sex"].str.lower().str.strip()
    df["Sex"] = pd.Categorical(df["Sex"], categories=sex_categories)

    idx = pd.MultiIndex.from_product(
        [pclass_categories, sex_categories, age_labels],
        names=["Pclass", "Sex", "age_group"]
    )

    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"], observed=False)
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum")
        )
        .reindex(idx, fill_value=0)
        .reset_index()
    )

    grouped["survival_rate"] = grouped.apply(
        lambda row: round(row["n_survivors"] / row["n_passengers"], 2)
        if row["n_passengers"] > 0 else 0,
        axis=1,
    )

    return grouped


def visualize_demographic(summary_df: pd.DataFrame):
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

    fig.show()


def family_groups(df: pd.DataFrame) -> pd.DataFrame:
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


def visualize_families(summary_df: pd.DataFrame):
    fig = px.scatter(
        summary_df,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        size="n_passengers",
        hover_data=["min_fare", "max_fare"],
        title="Average Ticket Fare by Family Size and Passenger Class",
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
            "Pclass": "Passenger Class",
        }
    )
    fig.show()


def last_names(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["last_name"] = df["Name"].str.split(",").str[0].str.strip()
    return df["last_name"].value_counts()


if __name__ == "__main__":
    FILE_PATH = "titanic.csv"  # Change as needed

    df = pd.read_csv(FILE_PATH)

    print("Exercise 1: Survival Demographics Summary")
    demo_summary = survival_demographics(df)
    print(demo_summary)

    print("\nVisualizing Survival Demographics...")
    visualize_demographic(demo_summary)

    print("\nExercise 2: Family Size and Wealth Summary")
    fam_summary = family_groups(df)
    print(fam_summary)

    print("\nVisualizing Family Groups...")
    visualize_families(fam_summary)

    print("\nTop Last Names and Counts")
    name_counts = last_names(df)
    print(name_counts.head(15))
