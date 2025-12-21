import os
from io import BytesIO
from PIL import Image
import base64
import requests
from pathlib import Path
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, TypedDict
from langchain.messages import AnyMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


# helper
def get_text(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

def make_unique_dir(base: str, suffix_format: str = "{}"):
    path = Path(base)
    counter = 1
    while path.exists():
        path = Path(f"{base}_{suffix_format.format(counter)}")
        counter += 1
    path.mkdir(parents=True)
    return path

def generate_image(prompt: str, model: str) -> dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": "16:9"
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    if result.get("choices"):
        message = result["choices"][0]["message"]
        if message.get("images"):
            dir_path = make_unique_dir("slides", suffix_format="{:02d}")
            for idx, image in enumerate(message["images"]):
                image_url = image["image_url"]["url"]
                print(f"Generated image: {image_url[:50]}...")
   
                image_data = image_url.split(",", 1)[1]
                image_data = base64.b64decode(image_data)
                with BytesIO(image_data) as image_buffer:
                    image = Image.open(image_buffer)
                    image.save(f"{dir_path}/output_{idx}.png", format="PNG")
    return result

def generate_text(prompt: str, model: str) -> dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]      
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


PLANNER_PROMPT = """
    You are a planner to make slides.
    You are given texts and you need to plan for making slides.
    1. figure out the main topic of the texts.
    2. decompose the texts into with their semantic meaning.
    3. make plans to generate slide images for each semantic meaning.
    Note: 
    1. A slide has only one sub topic. 
    2. slides should have texts some extent to be able to explain the sub topic.
    Texts: {texts}
   
    """

GENERATE_PROMPT = """
    You are a slide maker.
    You are given a plan and you need to generate slide images.
    Plan: {plan}
   
    """

# nodes
def planner(state: GraphState) -> GraphState:
    messages = state["messages"][-1].content
    res = generate_text(PLANNER_PROMPT.format(texts=messages), model="google/gemini-3-pro-preview")
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}

def generate_slide(state: GraphState) -> GraphState:
    plan = state["messages"][-1].content
    res = generate_image(GENERATE_PROMPT.format(plan=plan), model="google/gemini-2.5-flash-image")
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}


# graph
graph = StateGraph(GraphState)
graph.add_node("planner", planner)
graph.add_node("generate_slide", generate_slide)
graph.add_edge(START, "planner")
graph.add_edge("planner", "generate_slide")
graph.add_edge("generate_slide", END)


from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    sample = get_text("demo/sample.txt")
  
    config={"configurable": {"thread_id": "1"}}
    for chunk in app.stream({"messages": [{"role": "user", "content": sample}]}, stream_mode="updates", config=config):
        print(chunk)