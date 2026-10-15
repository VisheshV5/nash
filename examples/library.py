"""Ask Regret what to do: python examples/library.py <bundle dir or name> examples/state.json"""

import json
import sys
from pathlib import Path

from regret import RegretAgent

agent = RegretAgent.load(sys.argv[1])
decision = agent.act(json.loads(Path(sys.argv[2]).read_text()), explain=True, seed=0)
print(decision.strategy)
print(decision.recommended)
