from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from operator import add

class AgentState(TypedDict):
    """
    State for the Job Fit Agent LangGraph workflow.
    """
    jd_text: str
    resume_text: str
    
    # Extracted from inputs
    required_skills: list[str]
    candidate_skills: list[str]
    
    # Match state
    matched_skills: dict[str, str]
    missing_skills: list[str]
    
    # Conversation state
    messages: Annotated[Sequence[BaseMessage], add]
    current_question_index: int
    sub_question_index: int
    skills_to_assess: list[str]
    
    # Assessment state
    assessed_proficiency: dict[str, str] # skill -> "Low", "Medium", "High"
    assessment_complete: bool
    
    # Final output
    final_gaps: list[str]
    learning_plan: str
