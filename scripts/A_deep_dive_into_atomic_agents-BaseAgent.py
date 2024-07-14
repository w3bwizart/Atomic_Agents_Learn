# Install the necessary packages
# $pip install atomic-agents openai instructor Groq dotenv

# we need to be able to load an API_key from a .env file 
from dotenv import load_dotenv
# we need the os to get to out environment variables
import os
# we need to be able to provide a date
import datetime
# we need instructor to interact with the model
import instructor
# we need groq because it's free and easy to setup
from groq import Groq

# we need the AgentMemory because this is the memory of the agent
from atomic_agents.lib.components.agent_memory import AgentMemory
# we need the BaseAgent because it's the core of the framework 
# we need the BaseAgentConfig to create the configuration for the BaseAgent
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
# we need the SystemPromptInfo to create our system prompt
# we need the SystemPromptGenerator to generate the system prompt based on the SystemPromptInfo
# we need the SystemPromptContextProviderBase to create a context provider 
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptInfo, SystemPromptContextProviderBase
# we need the console to be able to print and have a peek into the magic
from rich.console import Console
# we need the console for obvious reasons
console = Console()
# we need to load the environment variables from the .env file
load_dotenv()

# Define the SystemPromptInfo object
system_prompt_info = SystemPromptInfo(    
    background=[
        'This assistant is a general-purpose AI designed to be helpful and friendly.',
    ],
    steps=[
        'Understand the user input.',
        'Reason about the input.',
        'Respond to the user.',
    ],
    output_instructions=[
        'Provide helpful and relevant information to assist the user.',
        'Be friendly and respectful in all conversations.',
        'Always use the available additional information and context to enhance the response',
    ],
)

# Let's extend the SystemPromptContextProviderBase
class CurrentDateContextProvider(SystemPromptContextProviderBase):
    def __init__(self, format: str = '%Y-%m-%d %H:%M:%S', **kwargs):
        super().__init__(**kwargs)
        self.format = format

    def get_info(self) -> str:
        return f'Date: {datetime.datetime.now().strftime(self.format)}' # **Remember** 'title' is from the base class we are extending.
    
# Create an instance of the new CurrentDateContextProvider, 
provider = CurrentDateContextProvider(title='Datetime')

# Add the new CurrentDateContextProvider to the SystemPromptInfo
system_prompt_info.context_providers = {
    'date': CurrentDateContextProvider(
        title='Datetime Context Provider', 
        format='%Y-%m-%d %H:%M:%S'
    )
}
# define the SystemPromptGenerator with the SystemPromptInfo object
system_prompt_generator = SystemPromptGenerator(system_prompt_info)

# Define initial memory with a greeting message from the assistant
initial_memory_message = [
    {'role': 'assistant', 
     'content': 'How do you do and what can I do for you today?',
     'tool_message': 'no tool used',
     'tool_id': None
    } 
]
# Initialize the agent memory to store conversation history
agent_memory = AgentMemory()

# Load the initial memory into the agent memory
agent_memory.load(initial_memory_message)

# get your API_Key from your enviroment variable if it's not there add your API_Key here 
API_KEY = ''
if not API_KEY:
    # get the environments variable
    API_KEY = os.getenv('GROQ_API_KEY') # OPENAI_API_KEY, OLLAMA_API_KEY, ...
    
if not API_KEY:
    raise ValueError('API key is not set. Please set the API key as a static variable or in an environment variable.')

client = instructor.from_groq(
    Groq(api_key=API_KEY,),
    mode=instructor.Mode.TOOLS
)

# configure the agent
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        system_prompt_generator=system_prompt_generator,
        model='llama3-70b-8192', # Specify the model you want to use
        memory=agent_memory
    )
)

# Main loop for testing the chat agent
# take the 1st message from the agentMemory and prefix with the role: Agent
console.print(f'Agent: {initial_memory_message[0]["content"]}')

while True:
    # get the user inpput and prefix with the role: User
    user_input = input('User: ')
    if user_input.lower() in ['/exit', '/quit']:
        print('Exiting chat, see you later ...')
        break
    
    # run the agent and asign the user input to the conversation
    response = agent.run(agent.input_schema(chat_message=user_input))
    print(f'Agent: {response.chat_message}')