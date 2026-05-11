import streamlit as st

st.set_page_config(page_title="Your Prep Plan", page_icon="🎯", layout="wide")

if "app_state" not in st.session_state or not st.session_state.app_state:
    st.warning("No assessment data found. Please start an assessment first.")
    if st.button("Go Back"):
        st.switch_page("app.py")
    st.stop()

st.title("🎯 Your Structured Prep Plan")

plan_markdown = st.session_state.app_state.get("learning_plan", "")

st.markdown(plan_markdown)

st.write("---")
if st.button("Start New Assessment"):
    st.session_state.app_state = None
    st.session_state.chat_history = []
    st.switch_page("app.py")
