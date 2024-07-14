# For a detailed explanation of this code checkout the notebook:
# /notebooks/A_deep_dive_into_atomic_agents_BaseAgent.ipynb

from dotenv import load_dotenv
import os
import datetime
import instructor
from groq import Groq

from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptGenerator,
    SystemPromptInfo,
    SystemPromptContextProviderBase,
)
from rich.console import Console

console = Console()
load_dotenv()

system_prompt_info = SystemPromptInfo(
    background=[
        "This assistant is a general-purpose AI designed to be helpful and friendly.",
    ],
    steps=[
        "Understand the user input.",
        "Reason about the input.",
        "Respond to the user.",
    ],
    output_instructions=[
        "Provide helpful and relevant information to assist the user.",
        "Be friendly and respectful in all conversations.",
        "Always use the available additional information and context to enhance the response",
    ],
)


class CurrentDateContextProvider(SystemPromptContextProviderBase):
    def __init__(self, format: str = "%Y-%m-%d %H:%M:%S", **kwargs):
        super().__init__(**kwargs)
        self.format = format

    def get_info(self) -> str:
        return f"Date: {datetime.datetime.now().strftime(self.format)}"


provider = CurrentDateContextProvider(title="Datetime")

system_prompt_info.context_providers = {
    "date": CurrentDateContextProvider(
        title="Datetime Context Provider", format="%Y-%m-%d %H:%M:%S"
    )
}
system_prompt_generator = SystemPromptGenerator(system_prompt_info)

initial_memory_message = [
    {
        "role": "assistant",
        "content": "How do you do and what can I do for you today?",
        "tool_message": "no tool used",
        "tool_id": None,
    }
]
agent_memory = AgentMemory()

agent_memory.load(initial_memory_message)

API_KEY = ""
if not API_KEY:
    API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in an environment variable."
    )

client = instructor.from_groq(
    Groq(
        api_key=API_KEY,
    ),
    mode=instructor.Mode.TOOLS,
)

agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        system_prompt_generator=system_prompt_generator,
        model="llama3-70b-8192",
        memory=agent_memory,
    )
)

console.print(f'Agent: {initial_memory_message[0]["content"]}')

while True:
    user_input = input("User: ")
    if user_input.lower() in ["/exit", "/quit"]:
        print("Exiting chat, see you later ...")
        break

    response = agent.run(agent.input_schema(chat_message=user_input))
    print(f"Agent: {response.chat_message}")
