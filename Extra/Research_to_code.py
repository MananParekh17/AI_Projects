from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.arxiv_toolkit import ArxivToolkit
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()



agent1 = Agent(
    name="Researcher",
    tools=[ArxivToolkit(read_arxiv_papers=True)],
    role="Extract and implement model architectures from research papers",
    instructions=(
        "Find the specified research paper on arXiv, provide a summary of the paper, "
        "extract the model architecture or methodology described, "
        "and implement it as Python code using the PyTorch framework. "
        "Include the link to the paper and ensure code correctness."
    ),
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    show_tool_calls=True
)

# Query the agent with a detailed task
agent1.print_response(
    "Search for the paper 'Attention is All You Need' by Ashish Vaswani et al. "
    "Summarize the paper, extract the Transformer model architecture, "
    "and implement it in Python using PyTorch. Provide complete code for the implementation.",
    markdown=True,
)
