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

class AsyncAzureOpenAI(AzureOpenAI):
    """Async variant stub mirroring AzureOpenAI interface for tests."""
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass

    class chat:
        class completions:
            @staticmethod
            async def create(*args, **kwargs):
                class Choice:
                    def __init__(self):
                        self.message = type('msg', (), {'content': ''})
                return type('Resp', (), {'choices': [Choice()]})()

class OpenAIError(Exception):
    pass
