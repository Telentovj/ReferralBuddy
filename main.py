import streamlit as st
import streamlit.components.v1 as components
from models import *

st.set_page_config(layout="wide")

with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

if "currentPage" not in st.session_state:
    st.session_state.currentPage = "main_page"

main_page = st.empty()
insight_page = st.empty()

# For changing pages
def change_page(page):
    st.session_state.currentPage = page


# Main page
if st.session_state.currentPage == "main_page":
    main_page = st.container()
    with main_page:
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        col2.image("logo.png", use_column_width=True)
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], key="enabled")
        if uploaded_file:
            st.session_state.df = read_and_process_file("IS4250JanFeb2023.csv")
            insight1, insight2, insight3 = st.columns([1, 0.5, 1])
            insight = insight2.button(
                "Click here to focus on the insights that has be found!",
                on_click=change_page,
                args=("insight_page",),
            )



# Insight Page
if st.session_state.currentPage == "insight_page":
    insight_page = st.container()
    with insight_page:
        df = st.session_state['df']
        eda_expander = st.expander("Data Exploration")
        eda_expander.pyplot(plot_eda(df))
        arima_expander = st.expander("Arima Model Prediction")
        arima_expander.pyplot(initialise_arima(df))