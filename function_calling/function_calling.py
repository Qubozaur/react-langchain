from typing import List, Optional

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
def add_numbers(x: int, y: int) -> int:
    """Return the sum of two integers."""
    return x + y


@tool
def to_uppercase(text: str) -> str:
    """Return the given text in uppercase."""
    return text.upper()


@tool
def reverse_string(string: str) -> str:
    """Returns reversed string."""
    return string[::-1]


def get_default_tools() -> List[BaseTool]:
    """Return the default set of tools used by the demo."""
    return [
        get_text_length,
        multiply,
        add_numbers,
        to_uppercase,
        reverse_string,
    ]

def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    """Helper to look up a tool by name."""
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


def run_question(question: str, tools: Optional[List[BaseTool]] = None) -> str:
    """Run a single question and return the LLM's final answer as text.
    Ask the LLM a question and let it decide which tools to call.

    Args:
        question: Natural language question for the LLM.
        tools: Optional custom list of tools. If not provided, a default
            set of tools (get_text_length, multiply, add_numbers,
            to_uppercase, reverse_string) will be used.
    """
    print(f"\nQuestion: {question}")

    if tools is None:
        tools = get_default_tools()

    llm = ChatOpenAI(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. "
                "You can use tools to get the length of text, add or multiply two numbers, "
                "convert text to uppercase, or reverse a string. "
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

        answer = ai_message.content
        print(f"Answer: {answer}")
        return answer


def chat_loop(tools: Optional[List[BaseTool]] = None) -> None:
    """
    Simple interactive loop so you can experiment with tool-calling.

    Type 'exit', 'quit', or press Ctrl+C to stop.
    """
    if tools is None:
        tools = get_default_tools()

    print("\nStarting interactive tool-calling chat.")
    print("Type your question and press Enter (or 'exit' to quit).\n")

    try:
        while True:
            user_input = input("> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            if not user_input:
                continue

            run_question(user_input, tools=tools)
    except KeyboardInterrupt:
        print("\nInterrupted, exiting chat.")


if __name__ == "__main__":
    print("Hello LangChain Tools!")
    run_question("What is the length of the word 'DOG'?")
    run_question("What is 7 times 13?")
    run_question("What is 'hello world :P' in reverse?")