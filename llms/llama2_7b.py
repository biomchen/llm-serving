import torch
from transformers import pipeline, AutoTokenizer

from utils import Configs


class Llama2_7b(Configs):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLAMA2_7B_MODEL)
        self.model = pipeline(
            self.LLAMA_TASK,
            model=self.LLAMA2_7B_MODEL,
            torch_dtype=torch.float16,
            device=self.LLAMA2_7B_CUDA
        )

    def get_response(self, prompt):
        results = []
        responses = self.model(
            prompt,
            do_sample=self.LLAMA_DO_SAMPLE,
            top_k=self.LLAMA_TOP_K,
            num_return_sequences=self.LLAMA_NUM_RETURN_SEQUENCE,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.LLAMA2_7B_MAX_LEN
        )
        for response in responses:
            print(f"Result: {response['generated_text']}")
            results.append(response['generated_text'])
        return responses[0]['generated_text']
    