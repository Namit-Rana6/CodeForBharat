# main.py
from crewai import Task, Crew
from agents import loader_agent, eda_agent, insight_agent, qa_agent
import os
import pandas as pd

def main():
    print("📊 Welcome to the CrewAI Data Analyst")
    
    # Check for default dataset
    default_file = "data/titanic.csv"
    if os.path.exists(default_file):
        print(f"🎯 Default dataset available: {default_file}")
        choice = input("📂 Press Enter to use default dataset, or enter custom file path: ").strip()
        file_path = choice if choice else default_file
    else:
        file_path = input("📂 Enter the path to your CSV file: ")

    if not os.path.exists(file_path):
        print("❌ File not found.")
        if not os.path.exists(default_file):
            print("💡 Tip: You can download the sample Titanic dataset by running 'python download_titanic.py'")
        return

    print("✅ File found. Beginning analysis...")
    
    # Pre-load the data to provide context to agents
    print("📂 Loading and analyzing dataset...")
    df = pd.read_csv(file_path)
    
    # Generate comprehensive dataset summary
    basic_info = f"""
Dataset Overview:
- File: {file_path}
- Shape: {df.shape[0]} rows, {df.shape[1]} columns
- Columns: {list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Missing Values Analysis:
{df.isnull().sum().to_string()}

Basic Statistics for Numeric Columns:
{df.describe().to_string()}

Sample Data (First 5 Rows):
{df.head().to_string()}

Additional Info:
- Total Missing Values: {df.isnull().sum().sum()}
- Duplicate Rows: {df.duplicated().sum()}
- Memory Usage: {df.memory_usage(deep=True).sum()} bytes
"""

    # Task 1: Data Loading Analysis
    task1 = Task(
        description=f"""
        Analyze this dataset summary and provide insights about data loading and structure:
        
        {basic_info}
        
        Focus on:
        - Dataset structure assessment
        - Data type appropriateness 
        - Missing data patterns
        - Overall data quality
        """,
        expected_output="Comprehensive data loading analysis with structure insights and quality assessment",
        agent=loader_agent
    )

    # Task 2: Exploratory Data Analysis
    task2 = Task(
        description=f"""
        Perform exploratory analysis based on this dataset information:
        
        {basic_info}
        
        Analyze:
        - Distribution patterns in the data
        - Relationships between variables
        - Notable trends or anomalies
        - Statistical insights from the data
        
        Provide detailed EDA findings without executing code.
        """,
        expected_output="Detailed exploratory data analysis with insights about distributions, correlations, and patterns",
        agent=eda_agent
    )

    # Task 3: Business Insights
    task3 = Task(
        description=f"""
        Generate actionable business insights from this dataset analysis:
        
        {basic_info}
        
        Provide:
        - Top 5-7 key findings
        - Business implications
        - Data quality recommendations
        - Suggested next steps for analysis
        """,
        expected_output="Strategic insights and actionable recommendations based on the dataset analysis",
        agent=insight_agent
    )

    # Run crew sequentially
    crew = Crew(
        agents=[loader_agent, eda_agent, insight_agent],
        tasks=[task1, task2, task3],
        process="sequential",
        verbose=True
    )

    crew.kickoff()

    # Follow-up questions loop
    print("\n💬 Ask questions about the dataset (type 'exit' to quit)")
    while True:
        q = input("🧠 Your question: ")
        if q.lower() in ["exit", "quit"]:
            break

        task_q = Task(
            description=f"""
            Answer this question about the dataset: {q}
            
            Dataset context:
            {basic_info}
            
            Provide a detailed, analytical answer based on the dataset information provided.
            """,
            expected_output="Detailed answer with supporting analysis based on the dataset context",
            agent=qa_agent
        )
        task_q.run()

if __name__ == "__main__":
    main()
