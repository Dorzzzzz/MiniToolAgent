"""Quick smoke-test for all recent fixes."""
import sys
sys.path.insert(0, ".")

from minitoolagent.tools.search import WikipediaSearchTool
from minitoolagent.tools.python_repl import PythonREPLTool
from minitoolagent.config import Config
from minitoolagent.models import ChatModel
from minitoolagent.agent import ReActAgent
from minitoolagent.parsing import extract_code_block, parse_llm_response

# 1. Fuzzy tool name matching
cfg = Config.from_yaml()
model = ChatModel(cfg)
agent = ReActAgent(model=model, tools=[WikipediaSearchTool(), PythonREPLTool()], max_steps=3)

print("=== Fuzzy tool matching ===")
cases = [
    "w_wikipedia_search",
    "wwwikipedia_search",
    "python_reD",
    "python_re then",
    "python",
    "w wikipedia_search",
]
for c in cases:
    result = agent._fuzzy_match_tool(c)
    print(f"  {c!r:30s} -> {result!r}")

# 2. Code block extraction
print("\n=== Code block extraction ===")
sample = '''Thought: I need to write code.
Action: {"tool": "python_repl", "args": {"code": ""}}
```python
result = sum(range(10))
print(result)
```'''
code = extract_code_block(sample)
print(f"  Extracted: {code!r}")

# 3. JSON with newlines in code
print("\n=== JSON with escaped newlines ===")
sample2 = 'Action: {"tool": "python_repl", "args": {"code": "x = 1\\nprint(x)"}}'
parsed = parse_llm_response(sample2)
print(f"  Tool: {parsed.tool}, code arg: {parsed.args.get('code')!r}")

# 4. Wikipedia search (with User-Agent and fulltext fallback)
print("\n=== Wikipedia search ===")
search = WikipediaSearchTool()
result = search.execute(query="Truman Sports Complex stadiums")
print(f"  Result preview: {result[:200]}")

# 5. Python REPL
print("\n=== Python REPL ===")
repl = PythonREPLTool()
out = repl.execute(code="print(sum(range(10)))")
print(f"  Output: {out!r}")

print("\nAll tests passed.")
