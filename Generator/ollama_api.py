import time
from langchain_ollama import ChatOllama


class OllamaChatTool:
    def __init__(self, args):
        self.model_name = args.model
        self.args = args

        if args.base_url:
            self.base_url = args.base_url
            self.client = ChatOllama(base_url=self.base_url, model=self.model_name)
        else:
            self.client = ChatOllama(model=self.model_name)
        return

    def generate(self, prompt, system_instruction='You are a helpful AI bot.', isrepeated=0.0, response_mime_type=None):
        temperature = self.args.temperature + isrepeated if isrepeated > 0.0 else self.args.temperature
        error = 3
        
        while error > 0:
            try:
                response = self.client.invoke(
                    input='\n'.join(prompt),
                    temperature=temperature,
                    seed=42,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                return response['response']
            
            except Exception as e:
                print(f'Ollama encountered an error: {str(e)}')
                error -= 1
                time.sleep(2.0)
                
        return None
