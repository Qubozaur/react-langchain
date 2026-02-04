from shlex import join
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
    """Returns the lenght of a text bby characters"""
    text = text.strip("'\n").strip('"')  # striping just in case
    return len(text)


if __name__ == "__main__":
    print("Hello react langchain :P ")
    tools = [get_text_length]

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
    intermediate_steps = []
    agent = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: x.get("agent_scratchpad", ""),
    } | prompt | llm

    llm_output = agent.invoke(
        {
            "input": "What is the length of 'DOG' in characters ?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    output_text = getattr(llm_output, "content", str(llm_output))
    print(output_text)

    agent_step: Union[AgentAction, AgentFinish] = parse_agent_step(output_text)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print(f"{observation}")
        intermediate_steps.append((agent_step, str(observation)))


    print("no kurwa mać, mail z polibudy mi psuje contribuitons")


