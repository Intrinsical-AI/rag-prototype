# RepoGPT Eval Repo

## Usage

```python
from client import build_api_client
from config import load_config

cfg = load_config({"TIMEOUT_S": "15"})
client = build_api_client("https://api.example.com")
```
