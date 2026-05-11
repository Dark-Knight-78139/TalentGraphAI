import json
import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from backend.agents.state import AgentState
from backend.services.llm import extract_skills_with_llm, get_llm
from backend.services.nlp import match_skills

def extract_and_match(state: AgentState):
    """
    Extracts skills from JD and Resume, and matches them.
    """
    jd_skills = extract_skills_with_llm(state["jd_text"], "job description")
    resume_skills = extract_skills_with_llm(state["resume_text"], "resume")
    
    match_result = match_skills(jd_skills, resume_skills)
    skills_to_assess = jd_skills
    
    return {
        "required_skills": jd_skills,
        "candidate_skills": resume_skills,
        "matched_skills": match_result["matched"],
        "missing_skills": match_result["missing"],
        "skills_to_assess": skills_to_assess,
        "current_question_index": 0,
        "sub_question_index": 0,
        "assessed_proficiency": {},
        "assessment_complete": False
    }

def conversational_assessment(state: AgentState):
    """
    Acts as the interviewer, evaluating answers and asking the next question.
    """
    llm = get_llm()
    messages = state["messages"]
    skills_to_assess = state.get("skills_to_assess", [])
    idx = state.get("current_question_index", 0)
    sub_idx = state.get("sub_question_index", 0)
    assessed = state.get("assessed_proficiency", {})
    
    if idx >= len(skills_to_assess):
        return {"assessment_complete": True}
        
    current_skill = skills_to_assess[idx]
    question_types = ["Cultural fit", "Technical fit", f"The specific required skill: {current_skill}"]
    
    if not messages or (len(messages) > 0 and getattr(messages[-1], 'type', '') != 'human'):
        current_focus = question_types[sub_idx]
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "Filter the input to extract only valid entities. Then, generate ONE concise high-quality interview question to evaluate the candidate on: {focus}. Output ONLY the question text. Do not include any other conversational filler."),
            ("human", "Generate the question for: {focus}")
        ])
        res = (question_prompt | llm).invoke({"focus": current_focus})
        content = re.sub(r'<think>.*?</think>', '', res.content, flags=re.DOTALL).strip()
        return {"messages": [AIMessage(content=content)]}
    
    last_answer = messages[-1].content
    updates = {}
    new_sub_idx = sub_idx
    new_idx = idx
    
    if sub_idx < 2:
        new_sub_idx = sub_idx + 1
        updates["sub_question_index"] = new_sub_idx
    else:
        recent_messages = messages[-6:]
        transcript = "\n".join([f"{'Agent' if getattr(m, 'type', '') == 'ai' else 'Candidate'}: {m.content}" for m in recent_messages])
        
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical assessor. Based on the candidate's conversation transcript regarding {skill}, evaluate their real overall proficiency and fit. Reply with EXACTLY one word: Low, Medium, or High."),
            ("human", "Transcript:\n{transcript}")
        ])
        eval_chain = eval_prompt | llm
        eval_res = eval_chain.invoke({"skill": current_skill, "transcript": transcript})
        proficiency = re.sub(r'<think>.*?</think>', '', eval_res.content, flags=re.DOTALL).strip()
        
        if proficiency not in ["Low", "Medium", "High"]:
            proficiency = "Medium"
            
        assessed[current_skill] = proficiency
        new_idx = idx + 1
        new_sub_idx = 0
        updates["assessed_proficiency"] = assessed
        updates["current_question_index"] = new_idx
        updates["sub_question_index"] = new_sub_idx

    if new_idx >= len(skills_to_assess):
        updates["assessment_complete"] = True
        return updates
        
    next_skill = skills_to_assess[new_idx]
    next_question_types = ["Cultural fit", "Technical fit", f"The specific required skill: {next_skill}"]
    next_focus = next_question_types[new_sub_idx]
    
    question_prompt = ChatPromptTemplate.from_messages([
        ("system", "Filter the input to extract only valid entities. Then, generate ONE concise high-quality interview question to evaluate the candidate on: {focus}. Output ONLY the question text. Do not include any other conversational filler."),
        ("human", "Generate the question for: {focus}")
    ])
    res = (question_prompt | llm).invoke({"focus": next_focus})
    content = re.sub(r'<think>.*?</think>', '', res.content, flags=re.DOTALL).strip()
    
    updates["messages"] = [AIMessage(content=content)]
    return updates

def generate_learning_plan(state: AgentState):
    """
    Identifies final gaps based on assessment, missing skills, and the candidate's resume,
    and generates a tailored learning plan dynamically using the LLM.
    """
    assessed = state.get("assessed_proficiency", {})
    missing = state.get("missing_skills", [])
    resume = state.get("resume_text", "")
    jd = state.get("jd_text", "")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI Career Coach. Your job is to create a structured prep plan to help the candidate bridge the gap between their resume and the job requirements.

Job Description Context:
{jd}

Candidate's Resume Context:
{resume}

Assessment Results (Proficiency level per skill):
{assessed}

Skills completely missing from the candidate's resume:
{missing}

Task:
Generate a highly structured and personalized learning plan. It MUST include:
1. An opening paragraph summarizing the candidate's overall readiness and the main gaps they need to fill.
2. A bulleted list of the exact skills/gaps they need to learn, specifically derived from their resume gaps and "Low/Medium" assessed skills.
3. For each bullet point, provide a precise time estimate required to master that skill.
4. For each bullet point, recommend one specific resource or approach to learn it.

Output ONLY clean, readable Markdown text. Do not output JSON. Do not include <think> tags in your final output."""),
        ("human", "Generate the tailored learning plan.")
    ])
    
    try:
        res = (prompt | llm).invoke({
            "jd": jd,
            "resume": resume,
            "assessed": json.dumps(assessed),
            "missing": json.dumps(missing)
        })
        
        content = re.sub(r'<think>.*?</think>', '', res.content, flags=re.DOTALL).strip()
        learning_plan_markdown = content
    except Exception as e:
        learning_plan_markdown = "An error occurred while generating your learning plan. Please try again."
        
    return {
        "final_gaps": [],
        "learning_plan": learning_plan_markdown
    }
