import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    """
    Initialize the LLM. 
    By default uses Groq, requires GROQ_API_KEY in environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY environment variable not set or invalid.")
    
    # We use qwen/qwen3-32b for fast/cheap reasoning, can be changed.
    return ChatGroq(model="qwen/qwen3-32b", temperature=0)

def extract_skills_with_llm(text: str, context: str = "job description") -> list[str]:
    """
    Uses LLM to robustly extract technical skills from text.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter. Extract a clean, comma-separated list of ONLY the technical skills, tools, and frameworks mentioned in the following {context}. Do not include soft skills. If none, return 'None'."),
        ("human", "{text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "text": text})
    import re
    content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    skills = [s.strip() for s in content.split(",") if s.strip().lower() != "none"]
    return skills
