import streamlit as st
import pandas as pd
from apputil import (
    survival_demographics,
    visualize_demographic,
    family_groups,
    last_names,
)

FILE_PATH = "titanic.csv"

# Read CSV once
df = pd.read_csv(FILE_PATH)

# --- Exercise 1: Demographic Analysis ---
st.header("Exercise 1: Survival Patterns")

demographic_summary = survival_demographics(df)
st.dataframe(demographic_summary)

st.write("Does age group have a stronger effect on survival rate for women than for men within each class?")
demographic_fig = visualize_demographic(demographic_summary)
st.plotly_chart(demographic_fig)

# --- Exercise 2: Family Groups Analysis ---
st.header("Exercise 2: Family Size and Wealth")

family_summary = family_groups(df)
st.dataframe(family_summary)

last_name_counts = last_names(df)
st.write("Which last names appeared most frequently, and how does this relate to variations in family size and survival?")
st.dataframe(last_name_counts)
