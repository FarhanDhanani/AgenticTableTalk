import time
from openai import OpenAI
import openai


class ChatGPTTool(object):
    def __init__(self,args, model_name, api_key_path, base_url=None):
            
        self.BASE_URL = base_url
        self.model_name = model_name

        self.args = args
        # chat
        self.API_SECRET_KEY = self.get_api_key(api_key_path)
        self.client = self.get_client(self.BASE_URL, 
                                      self.API_SECRET_KEY)
        return
    
    def get_client(self, base_url, secret_key):
        if base_url and secret_key:
            return OpenAI(api_key=secret_key, base_url=base_url)
        elif secret_key:
            return OpenAI(api_key=secret_key)
        else:
            raise ValueError("api_key not found. api_key is required parameter")
        return
    
    def get_api_key(self, api_key_path):
        if api_key_path is not None:
            return self.get_key(api_key_path)
        else:
            raise ValueError("api_key_path is required")
        return

    def get_key(self, key_path):
        try:
            with open(key_path, 'r') as f:
                key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f'Key file not found at {key_path}')
        return key
    
    def generate(self,prompt,system_instruction = 'You are a helpful AI bot.',isrepeated=0.0,response_mime_type=None):

        if isrepeated > 0.0:
            temperature = self.args.temperature + isrepeated
        else:
            temperature = self.args.temperature
        error = 3
        while error > 0:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": '\n'.join(prompt)}
                    ],
                    temperature=temperature,
                    seed=42,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    # response_format=response_mime_type,
                )

                break
            except openai.RateLimitError as r:
                print('openai rate-limited: ',r.__str__())
                error -= 1
                time.sleep(4.0)
            except openai.InternalServerError as r:
                print('openai crashed: ', r.__str__())
                error -= 1
                time.sleep(2.0)
            except openai.APITimeoutError as a:
                print('openai timeout: ', a.__str__())
                # error -= 1
                # time.sleep(2.0)
                raise UserWarning(f' openai timeout {a.__str__()}')
            except Exception as r:
                print('openai encountered an error: ',r.__str__())
                error -= 1
                time.sleep(2.0)
        output = resp.choices[0].message.content

        return output







