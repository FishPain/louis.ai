import os

def check_required_env_vars():
    required_env_vars = {
        "LLAMA_CLOUD_API_KEY": os.environ.get("LLAMA_CLOUD_API_KEY", None),
        "OPENAI_API_KEY" : os.environ.get("OPENAI_API_KEY", None),
        "LANGSMITH_TRACING": os.environ.get("LANGSMITH_TRACING", True),
        "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2", True),
        "LANGSMITH_ENDPOINT": os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "LANGSMITH_API_KEY": os.environ.get("LANGSMITH_API_KEY", None),
        "LANGSMITH_PROJECT": os.environ.get("LANGSMITH_PROJECT", "louisAI"),
        "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", None),
    }
    missing_env_vars = {key: value for key, value in required_env_vars.items() if value is None}
    
    # clear to prevent accidental printing
    required_env_vars = None

    if missing_env_vars:
        raise ValueError(f"Missing required environment variables: {missing_env_vars}")
    
    return missing_env_vars