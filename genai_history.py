class GenAIHistory:
    def __init__(self, api_key: str, model_name: str):
        self.data: any = None

        self.api_key = api_key
        self.model_name = model_name
