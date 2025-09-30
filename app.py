import streamlit as st
from titanic_analysis import (
    survival_demographics, visualize_demographic, family_groups, last_names
)

FILE_PATH = " Your file path goes here "

# --- Demographic Analysis ---
st.header("Titanic Survival Analysis by Demographic Groups")
demographic_summary = survival_demographics(FILE_PATH)
st.dataframe(demographic_summary)

st.write("Does age group have a stronger effect on survival rate for women than for men within each class?")
demographic_fig = visualize_demographic(demographic_summary)
st.plotly_chart(demographic_fig)

# --- Family Groups Analysis ---
st.header("Titanic Family Size and Wealth Analysis")
family_summary = family_groups(FILE_PATH)
st.dataframe(family_summary)

last_name_counts = last_names(FILE_PATH)
st.write("Which last names appeared most frequently, and how does this relate to variations in family size and survival?")
st.dataframe(last_name_counts)
