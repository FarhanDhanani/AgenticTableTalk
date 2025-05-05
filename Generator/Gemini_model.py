import google.generativeai as genai
from configs import GEMINI_KEYS
import random
import time
from google.api_core.exceptions import ResourceExhausted
class GeminiTool:
    def __init__(self , args):

        self.key_index = self.get_valid_key_idex(args.key_path)

        self.model_name = args.model

        self.args = args
        # self.key_index = GEMINI_KEYS.index(key)

        # Create the model
        # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
        self.safety_settings = [
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
            {
                "category": "HARM_CATEGORY_SEXUAL",
                "threshold": "BLOCK_NONE",
            },

        ]
        return
    
    def get_valid_key_idex(self, api_key_path):
        api_keys = self.get_api_key(api_key_path)

        for api_key in api_keys:
            if api_key not in GEMINI_KEYS:
                GEMINI_KEYS.append(api_key)
        
        return random.randint(0, len(GEMINI_KEYS) - 1)
    
    def get_api_key(self, api_key_path):
        if api_key_path is not None:
            return self.get_key(api_key_path)
        else:
            raise ValueError("api_key_path is required")
        return

    def get_key(self, key_path):
        try:
            with open(key_path, 'r') as f:
                keys = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f'Key file not found at {key_path}')
        return keys



    def generate(self ,prompt,system_instruction = 'You are a helpful AI bot.',isrepeated=0.0,response_mime_type=None):
        genai.configure(api_key=GEMINI_KEYS[self.key_index])
        generation_config = {
            "temperature": self.args.temperature + isrepeated if isrepeated > 0.0 else self.args.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": response_mime_type if response_mime_type != None else "text/plain",
        }
        model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings=self.safety_settings,
            generation_config=generation_config,
            system_instruction=system_instruction,
        )

        error = 3
        while error > 0:
            try:
                output = model.generate_content(prompt)
                result = output.text
                break
            except ValueError as v:
                raise UserWarning('unsafe input ' + v.__str__())
            except Exception as e:
                gemini_key_index = self.key_index
                print(GEMINI_KEYS[gemini_key_index], 'Error occurred.', e.__str__())
                gemini_key_index = random.randint(0, len(GEMINI_KEYS) - 1)
                genai.configure(api_key=GEMINI_KEYS[gemini_key_index])
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    safety_settings=self.safety_settings,
                    generation_config=generation_config,
                    system_instruction=system_instruction,
                )
                self.key_index = gemini_key_index
                print('Replace Gemini key with', GEMINI_KEYS[gemini_key_index])
                time.sleep(2.0)
                error -= 1
        if error <= 0:
            raise UserWarning('Gemini error')

        return result
