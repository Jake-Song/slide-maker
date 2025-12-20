import os
from io import BytesIO
from PIL import Image
import base64
import requests
from pathlib import Path
from langgraph.graph import StateGraph, START, END, add_messages
from typing import Annotated, TypedDict, List
import operator
from langchain.messages import AnyMessage, AIMessage
from langgraph.types import Send

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

def generate_structured_output(prompt: str, model: str, response_format: dict) -> dict:
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
        "response_format": response_format
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result

from pydantic import BaseModel, Field

class Slide(BaseModel):
    title: str = Field(
        description="title for this slide.",
    )
    text: str = Field(
        description="text for this slide.",
    )

class Slides(BaseModel):
    slides: List[Slide] = Field(
        description="Slides to present the document in a presentation.",
    )

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class WorkerState(TypedDict):
    plan: Slide
    slides: Annotated[list[Slide], operator.add]

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

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "presentation_slides",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "slides": {
                    "type": "array",
                    "description": "Slides to present the document in a presentation.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "title for this slide."
                            },
                            "text": {
                                "type": "string",
                                "description": "text for this slide."
                            }
                        },
                        "required": ["title", "text"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["slides"],
            "additionalProperties": False
        }
    }
}

# nodes
def planner(state: GraphState) -> GraphState:
    messages = state["messages"][-1].content
    res = generate_structured_output(
        PLANNER_PROMPT.format(texts=messages), 
        model="google/gemini-3-pro-preview", 
        response_format=response_format
    )
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}

def generate_slide(state: WorkerState) -> WorkerState:
    plan = state["plan"]
    res = generate_image(
        GENERATE_PROMPT.format(plan=plan), 
        model="google/gemini-2.5-flash-image"
    )
    return {"slides": [Slide(title=res["choices"][0]["message"]["content"])]}

def synthesizer(state: GraphState):
    """Synthesize full answer from queries"""

    # List of completed sections
    completed_slides = state["slides"]

    # Format completed section to str to use as context for final sections
    completed_slides = "\n\n---\n\n".join(completed_slides)

    return {"final_answer": completed_slides}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: GraphState):
    """Assign a worker to each section in the plan"""

    # Kick off query writing in parallel via Send() API
    return [Send("generate_slide", {"plan": p}) for p in state["plan"]]
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
    sample = get_text("texts/sample-2.txt")
  
    config={"configurable": {"thread_id": "1"}}
    for chunk in app.stream({"messages": [{"role": "user", "content": sample}]}, stream_mode="updates", config=config):
        print(chunk)
