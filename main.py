import re
from dotenv import load_dotenv
from langchain_core.tools import render_text_description
load_dotenv()
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Union
from langchain_core.agents import AgentAction, AgentFinish


def find_tool_by_name(tools, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found")


def format_agent_scratchpad(intermediate_steps):
    """Format intermediate steps into the agent scratchpad format."""
    if not intermediate_steps:
        return ""
    
    scratchpad = ""
    for step, observation in intermediate_steps:
        scratchpad += step.log
        scratchpad += f"\nObservation: {observation}\nThought: "
    
    return scratchpad


def parse_agent_step(output_text: str) -> Union[AgentAction, AgentFinish]:
    """Parse LLM text into an AgentAction or AgentFinish based on the ReAct-style format."""
    if "Final Answer:" in output_text:
        final_answer = output_text.split("Final Answer:")[-1].strip()
        return AgentFinish(return_values={"output": final_answer}, log=output_text)
    action_match = re.search(r"Action:\s*(.*)", output_text)
    action_input_match = re.search(r"Action Input:\s*(.*)", output_text, re.DOTALL)

    if not action_match or not action_input_match:
        raise ValueError(f"Could not parse agent step from output:\n{output_text}")

    tool = action_match.group(1).strip()
    tool_input = action_input_match.group(1).strip().strip('"').strip("'")

    return AgentAction(tool=tool, tool_input=tool_input, log=output_text)


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    text = text.strip("'\n").strip('"')  # stripping just in case
    return len(text)


@tool
def calculate(expression: str) -> float:
    """Evaluates a mathematical expression and returns the result.
    Example: calculate('2 + 2') returns 4.0"""
    try:
        # Safe evaluation - only allow basic math operations
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def reverse_string(text: str) -> str:
    """Reverses a given string.
    Example: reverse_string('hello') returns 'olleh'"""
    return text[::-1]


if __name__ == "__main__":
    print(" with langchain\n")
    tools = [get_text_length, calculate, reverse_string]

    template = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!
    Question: {input}   
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    agent = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_agent_scratchpad(x.get("intermediate_steps", [])),
    } | prompt | llm

    def run_agent(user_input: str, max_iterations: int = 10):
        """Run the ReAct agent until it reaches a final answer or max iterations."""
        intermediate_steps = []
        
        print(f"Question: {user_input}\n")
        
        for i in range(max_iterations):
            llm_output = agent.invoke({
                "input": user_input,
                "intermediate_steps": intermediate_steps,
            })
            
            output_text = getattr(llm_output, "content", str(llm_output))
            print(f"--- Step {i + 1} ---")
            print(output_text)
            print()
            
            agent_step = parse_agent_step(output_text)
            
            if isinstance(agent_step, AgentFinish):
                print(f"Final Answer: {agent_step.return_values['output']}")
                return agent_step.return_values['output']
            
            # Execute the action
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"Tool '{tool_name}' executed with input: {tool_input}")
            print(f"Observation: {observation}\n")
            
            intermediate_steps.append((agent_step, str(observation)))
        
        raise ValueError(f"Agent did not finish within {max_iterations} iterations")

    questions = [
        "What is the length of 'DOG' in characters?",
        "What is 15 * 23 + 7?",
        "Reverse the string 'hello world'",
    ]
    
    for question in questions:
        try:
            result = run_agent(question)
            print(f"\n{'='*50}\n")
        except Exception as e:
            print(f"Error: {e}\n")
            print(f"\n{'='*50}\n")
