import os
from io import BytesIO
from PIL import Image
import base64
import requests
import json
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
        # "image_config": {
        #     "aspect_ratio": "16:9"
        # }
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result

def save_image(result: dict, dir_path: Path, index: int):
    if result.get("choices"):
        message = result["choices"][0]["message"]
        if message.get("images"):
            image_url = message["images"][0]["image_url"]["url"]
            print(f"Generated image: {image_url[:50]}...")
   
            image_data = image_url.split(",", 1)[1]
            image_data = base64.b64decode(image_data)
            with BytesIO(image_data) as image_buffer:
                image = Image.open(image_buffer)
                image.save(dir_path / f"output_{index}.png", format="PNG")

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
    slides: Annotated[list[Slide], "slides to generate"]
    slides_with_index: Annotated[list[dict[int, str]], operator.add]

class WorkerState(TypedDict):
    slide: Annotated[Slide, "slide to generate"]
    slide_index: Annotated[int, "index of the slide to generate"]
    slides_with_index: Annotated[list[dict[int, str]], operator.add]

PLANNER_PROMPT = """
    You are a planner to make slides.
    You are given texts and you make slide decks based on the texts.
    1. figure out the main topic of the texts
    2. decompose the texts into with their semantic meaning
    3. make a title for each slide based on the semantic meaning
    4. make a text for each slide based on the semantic meaning
    5. return the slide titles and texts in a list of slides
    Texts: {texts}
   
    """

GENERATE_PROMPT = """
    You are a slide maker.
    You are given a slide title and text
    you need to generate slide images.
    Slide: {slide}
   
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
        model="google/gemini-3-flash-preview", 
        response_format=response_format
    )
    json_data = json.loads(res["choices"][0]["message"]["content"])
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])], "slides": json_data["slides"]}

def generate_slide(state: WorkerState) -> WorkerState:
    slide_index = state["slide_index"]
    slide = state["slide"]
    res = generate_image(
        GENERATE_PROMPT.format(slide=slide), 
        model="google/gemini-2.5-flash-image"
    )
   
    return {"slides_with_index": [{"index": slide_index, "response": res}]}

def synthesizer(state: GraphState) -> GraphState:
    """Synthesize full answer from queries"""
    slides_with_index = state["slides_with_index"]
    slides_with_index.sort(key=lambda entry: entry["index"])
    save_dir_path = make_unique_dir("slides", suffix_format="{:02d}")
    for entry in slides_with_index:
        save_image(entry["response"], save_dir_path, entry["index"])
    return {"messages": [AIMessage(content="Slides generated successfully")]}


# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: GraphState) -> list[Send]:
    """Assign a worker to each slide in the slides"""

    # Kick off query writing in parallel via Send() API
    return [Send("generate_slide", {"slide_index": i, "slide": p}) for i, p in enumerate(state["slides"])]

# graph
graph = StateGraph(GraphState)
graph.add_node("planner", planner)
graph.add_node("generate_slide", generate_slide)
graph.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
graph.add_edge(START, "planner")
graph.add_conditional_edges(
    "planner", 
    assign_workers, 
    ["generate_slide"]
)
graph.add_edge("generate_slide", "synthesizer")
graph.add_edge("synthesizer", END)

from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    sample = get_text("texts/sample_2.txt")
  
    config={"configurable": {"thread_id": "1"}}
    for chunk in app.stream({"messages": [{"role": "user", "content": sample}]}, stream_mode="updates", config=config):
        value = list(chunk.values())

        messages = value[0].get("messages") if value else None
        if messages:
            messages[-1].pretty_print()

