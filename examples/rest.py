"""The same over HTTP: python examples/rest.py http://localhost:8000 examples/state.json"""

import json
import sys
import urllib.request
from pathlib import Path

url, body = f"{sys.argv[1]}/act?seed=0", Path(sys.argv[2]).read_bytes()
request = urllib.request.Request(url, body, {"content-type": "application/json"})
decision = json.load(urllib.request.urlopen(request))
print(decision["strategy"])
print(decision["recommended"])
