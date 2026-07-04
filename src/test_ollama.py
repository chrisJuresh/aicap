import sys
import requests

host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
try:
    r = requests.get(host.rstrip("/") + "/api/tags", timeout=5)
    r.raise_for_status()
except Exception as e:
    print(f"Could not reach Ollama at {host}: {e}")
    raise SystemExit(1)

models = [m.get("name") for m in r.json().get("models", [])]
print("Ollama is running.")
print("Installed models:")
for model in models:
    print(" -", model)
