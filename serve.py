from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime

from utils.llms import LLM

class InputData(BaseModel):
    prompt: str

class OutputData(BaseModel):
    response: str

def load_model(model_id):
    if model_id:
        return LLM(model_id)
    else:
        return "Please put down correct model name."
        
def create_app(model):
    name, version = model.split('-')[0], model.split('-')[1:]
    print(f"{model} is selected.")
    print("Start to load the model ...")
    return FastAPI(
        title=f"Inference API for {name}",
        description="A fast and simple API to serve a variety of LLMs",
        version=f"{version}",
    )

model_id = input(f"Please provide the model id: ")
#device = input(f"Choose GPU: ")
app = create_app(model_id)

load_model_start = datetime.now()
model = load_model(model_id)
print("Model loading complete!")
print(f"Model load time: {datetime.now() - load_model_start}")

@app.post(f"/model", response_model=OutputData)
async def llm_response(request: Request, input_data: InputData):
    prompt = input_data.prompt
    response = model.get_response(prompt)
    return OutputData(response=response)
