class AzureOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                class Choice:
                    def __init__(self):
                        self.message = type('msg', (), {'content': ''})
                return type('Resp', (), {'choices': [Choice()]})()
class OpenAIError(Exception):
    pass
