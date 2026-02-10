from typing import Dict

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


"""
Chatbot with Conversation Memory
--------------------------------

An assistant that remembers conversation context and can hold natural dialogues.
Uses:
  - OpenAI GPT (via `ChatOpenAI`) for generating responses.
  - LangChain for managing conversation history with `RunnableWithMessageHistory`.

Run this file directly to start an interactive chat in your terminal:

    python -m conversation_memory.conversation_memory

Make sure the environment variable OPENAI_API_KEY is set.
"""


load_dotenv()


# Simple in-memory store mapping session_id -> chat history object.
_store: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (and lazily create) the chat history for a given session_id."""
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def create_chatbot_with_memory() -> RunnableWithMessageHistory:
    """
    Create a chatbot chain that remembers previous conversation turns.

    The memory is keyed by a configurable `session_id` passed at invoke time.
    """
    llm = ChatOpenAI(temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant. "
                    "You remember earlier parts of the conversation and use them "
                    "to provide coherent, context-aware answers."
                    "You are kind and want to help user"
                    "talk to then as like a friend would do."
                ),
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history


def chat(session_id: str = "default") -> None:
    """
    Start an interactive chat session with conversation memory.

    All turns in this process are stored under the provided `session_id`.
    """
    chatbot = create_chatbot_with_memory()

    print("Chatbot with Conversation Memory")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Bot: {response.content}\n")


if __name__ == "__main__":
    chat()

