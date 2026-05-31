import os
import logging
from crewai import Agent, Task, Crew, Process
from langchain_ollama import ChatOllama
from agents.tools import MilvusSearchTool

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalystAgent:
    def __init__(self):
        # Initialize LLM
        # Ensure OLLAMA_MODEL is set in .env or default to llama3
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0
        )
        
        self.search_tool = MilvusSearchTool()
        self._use_crewai = True

    def _direct_fallback_response(self, user_question: str) -> str:
        search_results = self.search_tool._run(user_question)
        if not search_results.strip():
            return "I could not find any indexed invoice data to answer that question."

        lower = search_results.lower()
        if lower.startswith("search backend unavailable") or lower.startswith("no indexed data"):
            return search_results

        prompt = (
            "You are an invoice analyst. Answer the user question using only the retrieved invoice context. "
            "If key information is missing, state that clearly. Cite source filenames when possible.\n\n"
            f"User question:\n{user_question}\n\n"
            "Retrieved context:\n"
            f"{search_results[:8000]}\n\n"
            "Answer:"
        )

        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content)

    def get_response(self, user_question: str) -> str:
        """
        Creates a CrewAI process to answer the user's question.
        """
        if not self._use_crewai:
            return self._direct_fallback_response(user_question)

        try:
            # 1. Define Agent
            invoice_analyst = Agent(
                role='Senior Invoice Analyst',
                goal='Accurately answer user questions based on invoice data.',
                backstory=(
                    "You are an expert financial analyst with access to a database of invoices. "
                    "Your job is to query the database to find relevant information and then "
                    "synthesize a clear, accurate answer for the user. "
                    "Always cite the invoice filename if possible."
                ),
                tools=[self.search_tool],
                llm=self.llm,
                verbose=False,
                allow_delegation=False
            )

            # 2. Define Task
            analysis_task = Task(
                description=(
                    f"The user has asked: '{user_question}'\n"
                    "1. Use the Invoice Search Tool to find relevant invoice data.\n"
                    "2. Analyze the retrieved data to answer the question.\n"
                    "3. If the information is partial, explain what is found and what is missing.\n"
                    "4. Provide a final answer."
                ),
                expected_output="A detailed answer to the user's question, citing sources.",
                agent=invoice_analyst
            )

            # 3. Create Crew
            crew = Crew(
                agents=[invoice_analyst],
                tasks=[analysis_task],
                process=Process.sequential,
                verbose=False
            )

            # 4. Kickoff
            result = crew.kickoff()
            return str(result)

        except Exception as e:
            logger.warning(f"CrewAI path failed, using direct fallback: {e}")
            self._use_crewai = False
            try:
                return self._direct_fallback_response(user_question)
            except Exception as fallback_error:
                logger.error(f"Error in Analyst Agent fallback: {fallback_error}")
                return (
                    "Sorry, I encountered an error analyzing your request: "
                    f"{str(fallback_error)}"
                )

if __name__ == "__main__":
    # Test
    agent = AnalystAgent()
    print(agent.get_response("What is the total of the last invoice?"))
