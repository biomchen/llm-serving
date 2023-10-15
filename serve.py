from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime

from utils import Configs

from llms.llama2_7b import Llama2_7b
from llms.llama2_13b import Llama2_13b
from llms.codellama_7b_python import CodeLlamaPython7b
from llms.codellama_13b_python import CodeLlamaPython13b

configs = Configs()
version = configs.VERSION
llms = configs.MODELS

class InputData(BaseModel):
    prompt: str

class OutputData(BaseModel):
    response: str

def select_llm_model(model):
    if not model:
        return "Please copy and paste the correct model name."
    if model == "llama2_7b_chat":
        return Llama2_7b()
    elif model == "llama2_13b_chat":
        return Llama2_13b()
    elif model == "codellama_7b_python":
        return CodeLlamaPython7b()
    elif model == "codellama_13b_python":
        return CodeLlamaPython13b()
    
def create_app(model, version):
    name = " ".join([x[0].upper() + x[1:] for x in model.split('_')])
    print(f"{name} is selected.")
    print("Start to load the model ...")
    return FastAPI(
        title=f"Inference API for {name}",
        description="A fast and simple API to serve a variety of LLMs",
        version=f"{version}",
    )

llm = input(f"Please select a model from {', '.join(llms)}: ")
app = create_app(llm, version)

load_model_start = datetime.now()
model = select_llm_model(llm)
print("Model loading complete!")
print(f"Model load time: {datetime.now() - load_model_start}")

@app.post(f"/{llm}", response_model=OutputData)
async def llm_response(request: Request, input_data: InputData):
    # Get the prompt from the input data
    prompt = input_data.prompt
    response = model.get_response(prompt)
    return OutputData(response=response)
