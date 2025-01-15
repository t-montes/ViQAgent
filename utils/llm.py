import google.generativeai as genai
import json
import time
import os

MAX_RETRIES = 20
RETRY_DELAY_START = 10
RETRY_DELAY_INCREASE = 5
current_retry_delay = RETRY_DELAY_START

safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class LLM:
    def __init__(self, model_name, system_prompt=None, json_schema=None, temperature=0.0, seed=None, api_key=None, log=print):
        if api_key is not None: genai.configure(api_key=api_key)
        genconf = {
            "temperature":temperature,
        }
        if json_schema is not None:
            genconf["response_mime_type"] = "application/json"
            genconf["response_schema"] = json_schema
        if seed is not None:
            genconf["seed"] = seed

        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=genconf,
            safety_settings=safe
        )
        self.json_schema = json_schema
        self.log = log

    def __call__(self, query):
        ctx = [query]
        r = self.model.generate_content(ctx)
        response = r.text
        if self.json_schema is not None: response = json.loads(response)
        else: response = response
        return response, (r.usage_metadata.prompt_token_count, r.usage_metadata.candidates_token_count)

class VLLM(LLM):
    def __call__(self, content_paths, query, retry_count=0):
        global current_retry_delay
        if isinstance(content_paths, str): content_paths = [content_paths]
        self.last_execution_files = []
        ctx = []
        try:
            for content_path in content_paths:
                if content_path.startswith("http"):
                    raise ValueError("URL download not implemented yet")
                else:
                    file = upload_file(content_path)
                    ctx.append(file)
                    self.last_execution_files.append(file)
                ctx.append(query)
            r = self.model.generate_content(ctx)
            response = r.text
            # Example of blocked-prompt error: ValueError: Invalid operation: The `response.parts` quick accessor requires a single candidate, but but `response.candidates` is empty. This appears to be caused by a blocked prompt, see `response.prompt_feedback`: block_reason: OTHER
            if self.json_schema is not None: response = json.loads(response)
            else: response = response
        except Exception as e:
            errortype = str(type(e).__name__)
            if errortype == "ResourceExhausted":
                if retry_count < MAX_RETRIES:
                    self.log(f"ResourceExhausted, retrying ({retry_count+1}/{MAX_RETRIES}) [{current_retry_delay}s]")
                    time.sleep(current_retry_delay)
                    current_retry_delay += RETRY_DELAY_INCREASE
                    return self(content_paths, query, retry_count+1)
                else:
                    exit(f"ResourceExhausted, max retries reached ({MAX_RETRIES})")
            else:
                raise e
        current_retry_delay = RETRY_DELAY_START
        return response, (r.usage_metadata.prompt_token_count, r.usage_metadata.candidates_token_count)

# note: try as much as possible that the names differ when they are different inputs; if not, it will lead to prev-content conflict
def upload_file(path):
    existing = { file.display_name:file for file in list_files() }
    file_name = os.path.basename(path)
    
    if file_name in existing:
        return existing[file_name]
    else:
        file = genai.upload_file(path=path)
        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = genai.get_file(file.name)
        if file.state.name == "FAILED":
            raise ValueError(f"Failed to upload file: {file.uri} ({file.display_name})")
        return file

def remove_files(files):
    if not hasattr(files, "__iter__"): files = [files]
    for file in files:
        genai.delete_file(file.name)

def list_files():
    return genai.list_files()

def flush_files():
    for file in list_files():
        genai.delete_file(file.name)
