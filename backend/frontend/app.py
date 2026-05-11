import streamlit as st
import os
import sys

# Ensure backend can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.agents.workflow import build_graph
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Job Fit & Skill Assessment Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI Job Fit Assessor & Learning Planner")
st.markdown("Upload a Job Description and a Resume. The AI will extract skills, assess proficiency conversationally, and generate a personalized learning plan.")

# Load Env Vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GROQ_API_KEY"):
    st.sidebar.warning("Please set GROQ_API_KEY in the `.env` file or environment.")
    st.stop()

# Initialize session state for Graph and Chat
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "app_state" not in st.session_state:
    st.session_state.app_state = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "jd" not in st.session_state:
    st.session_state.jd = ""
if "resume" not in st.session_state:
    st.session_state.resume = ""

if st.session_state.app_state and st.session_state.app_state.get("assessment_complete"):
    st.switch_page("pages/plan.py")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Inputs")
    import pypdf
    
    def extract_text_from_pdf(file_upload):
        if file_upload is not None:
            pdf_reader = pypdf.PdfReader(file_upload)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        return ""

    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    resume_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
    
    if st.button("Start Analysis"):
        if jd_file is None or resume_file is None:
            st.error("Please upload both the JD and the Resume as PDF files.")
        else:
            with st.spinner("Extracting text from PDFs and matching skills..."):
                jd_input = extract_text_from_pdf(jd_file)
                resume_input = extract_text_from_pdf(resume_file)
                
                st.session_state.jd = jd_input
                st.session_state.resume = resume_input
                
                # Initial state setup
                initial_state = {
                    "jd_text": jd_input,
                    "resume_text": resume_input,
                    "messages": [],
                    "sub_question_index": 0
                }
                
                # Run graph until it pauses for input
                result = st.session_state.graph.invoke(initial_state)
                st.session_state.app_state = result
                st.session_state.chat_history = result.get("messages", [])
                st.rerun()

with col2:
    st.subheader("2. Conversational Assessment")
    
    if st.session_state.app_state:
        # Display chat history
        for msg in st.session_state.chat_history:
            if getattr(msg, "type", "") == "ai":
                st.info(f"**Agent:** {msg.content}")
            else:
                st.success(f"**You:** {msg.content}")
                
        # Show input box for next answer (made larger as requested)
        with st.form("answer_form", clear_on_submit=True):
            user_input = st.text_area("Your Answer:", height=150)
            submitted = st.form_submit_button("Submit Answer")
            if submitted:
                if user_input:
                    with st.spinner("Evaluating..."):
                        current_state = st.session_state.app_state
                        current_state["messages"].append(HumanMessage(content=user_input))
                    
                    from backend.agents.nodes import conversational_assessment, generate_learning_plan
                    from backend.agents.workflow import should_continue
                    
                    # Manually advance the state machine for Streamlit MVP
                    current_state = {**current_state, **conversational_assessment(current_state)}
                    action = should_continue(current_state)
                    
                    if action == "generate_learning_plan":
                        current_state = {**current_state, **generate_learning_plan(current_state)}
                        current_state["assessment_complete"] = True
                    elif action == "conversational_assessment":
                        # It generated a new question
                        pass
                        
                    st.session_state.app_state = current_state
                    st.session_state.chat_history = current_state.get("messages", [])
                    st.rerun()
    else:
        st.write("Provide JD and Resume on the left and click 'Start Analysis'.")
