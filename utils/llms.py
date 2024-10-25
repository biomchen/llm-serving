import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from .configs import ConfigsLoader
from .constants import MODEL_PATH
from .prompts import SYS_PROMPT #SYS_PROMPT_W_SEARCH_RESULTS

class LLM(ConfigsLoader):

    def __init__(self, model_id):
        super().__init__()
        self.MODEL_ID = os.path.join(MODEL_PATH, model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = pipeline(
            self.TASK,
            model=self.MODEL_ID,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    def get_response(self, prompt):
        results = []
        responses = self.model(
            SYS_PROMPT + prompt,
            do_sample=self.DO_SAMPLE,
            top_k=self.TOP_K,
            num_return_sequences=self.NUM_RETURN_SEQUENCE,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.MAX_LENGTH
        )
        for response in responses:
            results.append(response['generated_text'])
        return results[0].split('[/INST] ')[-1]
    
    # def get_response_w_search(self, prompt):
    #     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     search_results = self.search(prompt)
    #     results = []
    #     responses = self.model(
    #         SYS_PROMPT_W_SEARCH_RESULTS.format(
    #             datetime=now, 
    #             prompt=prompt, 
    #             context=search_results
    #         ) + prompt + '[/INST]',
    #         # SYS_PROMPT + prompt + '[/INST]',
    #         do_sample=self.LLAMA_DO_SAMPLE,
    #         top_k=self.LLAMA_TOP_K,
    #         num_return_sequences=self.LLAMA_NUM_RETURN_SEQUENCE,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         max_length=self.LLAMA2_13B_MAX_LEN
    #     )
    #     for response in responses:
    #         print(f"Result: {response['generated_text']}")
    #         results.append(response['generated_text'])
    #     return responses[0]['generated_text'].split('[/INST]  ')[-1]