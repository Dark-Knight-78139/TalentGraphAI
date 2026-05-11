from langgraph.graph import StateGraph, END
from backend.agents.state import AgentState
from backend.agents.nodes import extract_and_match, conversational_assessment, generate_learning_plan

def should_continue(state: AgentState):
    if state.get("assessment_complete"):
        return "generate_learning_plan"
    else:
        # We check if we need to ask a question or evaluate.
        # If the last message is an AIMessage, it means we are waiting for user input.
        # So we interrupt and END for now. 
        # In a real deployed app, LangGraph checkpointing pauses execution.
        last_msg = state["messages"][-1] if state.get("messages") else None
        if last_msg and getattr(last_msg, "type", "") == "ai":
            return END
        return "conversational_assessment"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_and_match", extract_and_match)
    workflow.add_node("conversational_assessment", conversational_assessment)
    workflow.add_node("generate_learning_plan", generate_learning_plan)
    
    # Define edges
    workflow.set_entry_point("extract_and_match")
    workflow.add_edge("extract_and_match", "conversational_assessment")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "conversational_assessment",
        should_continue,
        {
            "generate_learning_plan": "generate_learning_plan",
            "conversational_assessment": "conversational_assessment",
            END: END
        }
    )
    
    workflow.add_edge("generate_learning_plan", END)
    
    # Compile
    return workflow.compile()
