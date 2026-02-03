from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool

@tool
def get_text_length(text:str) -> int:
    """Returns the lenght of a text bby characters"""
    return len(text)

if __name__ == "__main__":
    print("Hello react agents")
    print(get_text_length("abc"))