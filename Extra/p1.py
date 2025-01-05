from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.arxiv_toolkit import ArxivToolkit
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo



load_dotenv()

agent1 = Agent(tools=[ArxivToolkit(read_arxiv_papers=True)],
              model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
              show_tool_calls=True)

# agent2 = Agent(tools=[DuckDuckGo()],
#                model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
#                show_tool_calls=True)
# agent2.print_response("find the research paper 'https://arxiv.org/pdf/1706.03762v7' ", markdown=True)

agent1.print_response("Search arxiv papers for 'attention is all you need' by Ashish Vaswani and provide link of paper too and and inside that paper find the model architecture if available and make a model described in paper in python language with pytorch framework", markdown=True,)

# agent1.print_response("Search for the paper titled ['attention is all you need'] by [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin] on arXiv. Provide the link to the paper. Extract the model architecture or methodology described in the paper and implement it as a complete Python code using the PyTorch framework. Ensure the implementation follows the details provided in the paper as closely as possible", markdown=True,)

# and inside that paper find the model architecture if available and make a model descirbed in paper in python language with pytorch framework