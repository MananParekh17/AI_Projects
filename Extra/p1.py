from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.arxiv_toolkit import ArxivToolkit
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo



# load_dotenv()

# agent1 = Agent(name = "Miner",
#                tools=[ArxivToolkit(read_arxiv_papers=True)],
#                role="find the model architecture in asked research paper and make a model described in paper in python language with pytorch framework",
#                instructions="always include summary and link of the paper ",
#                model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
#                show_tool_calls=True)

# # agent2 = Agent(name = "answer",
# #                tools=[ArxivToolkit(read_arxiv_papers=True)],
# #                role="find the model architecture from paper and make a model described in paper in python language with pytorch framework",
# #                instructions="always include complete code in the output",
# #                model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
# #                show_tool_calls=True)

# # agent2 = Agent(tools=[DuckDuckGo()],
# #                model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
# #                show_tool_calls=True)
# # agent2.print_response("find the research paper 'https://arxiv.org/pdf/1706.03762v7' ", markdown=True)

# # combined_agent = Agent(name = "combined",
# #                        team=[agent1, agent2],
# #                        role="combine the two agents and provide the final answer",
# #                        instructions=["always include summary and link of the paper","always include complete code in the output"],
# #                        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
# #                        show_tool_calls=True)

# agent1.print_response("Search arxiv papers for 'attention is all you need' by Ashish Vaswani and then inside that paper find the model architecture and make a model described in paper in python language with pytorch framework", markdown=True,)


# # agent1.print_response("Search for the paper titled ['attention is all you need'] by [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin] on arXiv. Provide the link to the paper. Extract the model architecture or methodology described in the paper and implement it as a complete Python code using the PyTorch framework. Ensure the implementation follows the details provided in the paper as closely as possible", markdown=True,)

# # and inside that paper find the model architecture if available and make a model descirbed in paper in python language with pytorch framework

# Define the agent with precise instructions


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
