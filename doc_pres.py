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

from pydantic import BaseModel, Field
from typing import List

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

def generate_structured_output(doc: str, schema: BaseModel) -> dict:
    from google import genai
   
    client = genai.Client()

    prompt = """
    make slides for the following text.
    {text}
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt.format(text=doc),
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema.model_json_schema(),
        },
    )

    res = schema.model_validate_json(response.text)
    return res

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Schema for structured output to use in planning
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


BIRDS_EYE_VIEW_PROMPT = """
    You are given a document and you need to generate a birds eye view of the document.
    generate main topics and sub topics.
    generate summary of sections under sub topics.
    Document: {text}
    """

OUTLINE_PROMPT = """
    You are given a birds eye view of a document and you need to generate an outline for a presentation.
    generate main title and sub titles for each section.
    decribe a flow of the presentation.    
    Birds Eye View: {birds_eye_view}
    """

SLIDE_TEXT_PROMPT = """
    You are given an outline and you need to generate a slide text.
    generate text that is concise and to the point.
    Outline: {outline}
    """
SLIDE_IMAGE_PROMPT = """
    You are given slide texts and you need to generate slide images for each slide text.
    Slide Texts: {slide_texts}
    """

# nodes
def birds_eye_view(state: GraphState) -> GraphState:
    messages = state["messages"][-1].content
    res = generate_text(BIRDS_EYE_VIEW_PROMPT.format(text=messages), model="google/gemini-3-pro-preview")
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}

def outline(state: GraphState) -> GraphState:
    birds_eye_view = state["messages"][-1].content
    res = generate_text(OUTLINE_PROMPT.format(birds_eye_view=birds_eye_view), model="google/gemini-3-flash-preview")
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}

def slide_text(state: GraphState) -> GraphState:
    outline = state["messages"][-1].content
    res = generate_text(SLIDE_TEXT_PROMPT.format(outline=outline), model="google/gemini-3-flash-preview")
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}

def slide_image(state: GraphState) -> GraphState:
    slide_texts = state["messages"][-1].content
    res = generate_image(SLIDE_IMAGE_PROMPT.format(slide_texts=slide_texts), model="google/gemini-2.5-flash-image")
    return {"messages": [AIMessage(content=res["choices"][0]["message"]["content"])]}

# graph
graph = StateGraph(GraphState)
graph.add_node("birds_eye_view", birds_eye_view)
graph.add_node("outline", outline)
graph.add_node("slide_text", slide_text)
graph.add_node("slide_image", slide_image)
graph.add_edge(START, "birds_eye_view")
graph.add_edge("birds_eye_view", "outline")
graph.add_edge("outline", "slide_text")
graph.add_edge("slide_text", "slide_image")
graph.add_edge("slide_image", END)


from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    sample = get_text("demo/sample.txt")
  
    config={"configurable": {"thread_id": "1"}}
    for chunk in app.stream({"messages": [{"role": "user", "content": sample}]}, stream_mode="updates", config=config):
        value = list(chunk.values())
        value[0]["messages"][-1].pretty_print()
    # res = generate_structured_output(BIRDS_EYE_VIEW_PROMPT.format(text=sample), model="google/gemini-3-flash-preview", schema=Slides)
    # print(res["choices"][0]["message"]["content"])
