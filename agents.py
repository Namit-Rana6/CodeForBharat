# agents.py
import os
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# === Gemini API Key ===
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyDloWhFDYK4MGsnAz3MFFUVmUaYlLSeyC8")
os.environ["GOOGLE_API_KEY"] = api_key

# === Language model with specific configuration for CrewAI ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=2048,
    timeout=60,
    max_retries=2
)

# === Agents without tools (using pure reasoning) ===
loader_agent = Agent(
    role="Data Loader",
    goal="Analyze CSV dataset structure and provide comprehensive data overview",
    backstory="Expert data analyst specializing in dataset structure analysis and data quality assessment",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    memory=True
)

eda_agent = Agent(
    role="EDA Specialist", 
    goal="Perform exploratory data analysis and identify patterns in data",
    backstory="Statistical analyst expert in data exploration, pattern recognition, and visualization insights",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    memory=True
)

insight_agent = Agent(
    role="Insight Reporter",
    goal="Generate actionable business insights from data analysis results",
    backstory="Business intelligence specialist who translates data findings into strategic recommendations",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    memory=True
)

qa_agent = Agent(
    role="Follow-up Analyst",
    goal="Answer specific questions about dataset characteristics and findings",
    backstory="Data consultant expert at providing detailed answers about dataset properties and analysis results",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,
    memory=True
)
