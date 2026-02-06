from typing import List

from dotenv import load_dotenv
from langchain.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Return the length of a text in characters."""
    print(f"get_text_length enter with {text=}")
    # Strip quotes/newlines that might be added around the raw text
    text = text.strip("'\n").strip('"')
    return len(text)


@tool
def multiply(x: int, y: int) -> int:
    """Return the result of multiplying two integers."""
    return x * y

@tool
def reverse_string(string: str) -> str:
    """Returns reversed string"""
    return string[::-1]

def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """Helper to look up a tool by name."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


def run_question(question: str) -> None:
    """Ask the LLM a question and let it decide which tools to call."""
    print(f"\nQuestion: {question}")

    tools: List[BaseTool] = [get_text_length, multiply, reverse_string]

    llm = ChatOpenAI(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "You can use tools to get the length of text, multiply two numbers or to reverse string. "
                "Use tools when they help answer the question."
            )
        ),
        HumanMessage(content=question),
    ]

    while True:
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if tool_calls:
            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"Tool '{tool_name}' called with {tool_args=} -> {observation=}")

                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            continue

        print(f"Answer: {ai_message.content}")
        break


if __name__ == "__main__":
    print("Hello LangChain Tools!")
    run_question("What is the length of the word 'DOG'?")
    run_question("What is 7 times 13?")
    run_question("What is 'hello' in reverse?")
    print(all_)