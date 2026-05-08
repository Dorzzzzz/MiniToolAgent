from __future__ import annotations

# Shared system prompt

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful assistant that solves tasks step by step using available tools.

## Available Tools
{tool_descriptions}

## Response Format
At each step, you MUST respond in the following format:

Thought: <your reasoning about what to do next>
Action: <a JSON object specifying the tool call>

The Action MUST be a valid JSON object with exactly two keys:
- "tool": the name of the tool to call (one of: {tool_names})
- "args": a JSON object with the tool's parameters

Example:
Thought: I need to search for information about Albert Einstein.
Action: {{"tool": "wikipedia_search", "args": {{"query": "Albert Einstein"}}}}

Python calculation example:
Thought: I need to verify this calculation with code.
Action: {{"tool": "python_repl", "args": {{"code": "total = sum(i * i for i in range(1, 6))\\nprint(total)"}}}}

When you have enough information to answer the question, use the special "final_answer" tool:
Thought: I now have enough information to answer.
Action: {{"tool": "final_answer", "args": {{"answer": "your final answer here"}}}}

## Important Rules
1. Always use ONE tool call per step. Do NOT call multiple tools in one response.
2. Wait for the Observation after each Action before proceeding.
3. Keep your final answer concise and precise.
4. If a tool returns an error, analyze the error and try a different approach.
"""

# Search task prompt (HotpotQA)

SEARCH_TASK_PROMPT = """\
Answer the following question by searching for relevant information. \
This may require multiple search steps to gather all necessary facts. \
Break down complex questions into sub-questions and search for each part.

Question: {question}
"""

# Few-shot math task prompt (AIME / HMMT)

MATH_TASK_PROMPT = """\
Solve the following math competition problem step by step.

## Strategy
1. Analyze the problem type (number theory, algebra, geometry, combinatorics).
2. Plan your mathematical approach in Thought before writing any code.
3. Write Python code for exact computation — avoid floating-point when an exact \
answer is needed.
4. Use sympy for symbolic algebra; use fractions.Fraction for exact rationals.
5. Verify your result with a sanity check or alternate method before submitting.
6. Call final_answer with only the requested value (integer, or simplified fraction).

## Available Libraries
- sympy: symbols, solve, simplify, expand, factor, factorint, isprime, \
  Rational, sqrt, pi, cos, sin, tan, acos, asin, atan2, \
  binomial, factorial, gcd, lcm, geometry (Point, Triangle, Circle, Polygon)
- fractions: Fraction  (exact rational arithmetic, e.g. Fraction(2,3)+Fraction(3,4))
- math: factorial, comb, gcd, isqrt, log, sqrt, pi
- itertools, collections, statistics, random, re

Variables you define in one python_repl call are available in later calls \
within the same problem.

## Answer Format Rules
- Give the **exact** answer — never a decimal approximation unless the problem asks for one.
- Integers: output as plain integer (e.g. `588`).
- Fractions: output as `p/q` in lowest terms (e.g. `7/18`).
- Radicals: output as `a*sqrt(b)` or `a + b*sqrt(c)` (e.g. `10*sqrt(3)`).
- AIME answers are always integers 0–999.
- HMMT answers may be fractions, radicals, or expressions involving π.
- Use `sympy.simplify()` or `sympy.nsimplify()` to obtain a clean exact form.

## Examples

### Example 1 — Number Theory (sympy.factorint)
Problem: Find the number of positive divisors of 9!.

Thought: Compute 9!, find its prime factorization with sympy, then apply the \
divisor-count formula: product of (e_i + 1) over each prime p_i^e_i.
Action: {{"tool": "python_repl", "args": {{"code": "import math\\nfrom sympy import factorint\\nn = math.factorial(9)\\nfactors = factorint(n)\\nprint('9! =', n)\\nprint('Factorization:', factors)\\nnum_div = 1\\nfor exp in factors.values():\\n    num_div *= (exp + 1)\\nprint('Number of divisors:', num_div)"}}}}

Observation:
9! = 362880
Factorization: {{2: 7, 3: 4, 5: 1, 7: 1}}
Number of divisors: 160

Thought: The divisor count is 160.
Action: {{"tool": "final_answer", "args": {{"answer": "160"}}}}

### Example 2 — Algebra (sympy.solve, exact roots)
Problem: Find the sum of the real roots of 2x³ − 11x² + 17x − 6 = 0.

Thought: Use sympy.solve to get exact roots, filter to real ones, then sum.
Action: {{"tool": "python_repl", "args": {{"code": "from sympy import symbols, solve\\nx = symbols('x')\\nroots = solve(2*x**3 - 11*x**2 + 17*x - 6, x)\\nprint('All roots:', roots)\\nreal_roots = [r for r in roots if r.is_real]\\nprint('Real roots:', real_roots)\\nprint('Sum:', sum(real_roots))"}}}}

Observation:
All roots: [1/2, 2, 3]
Real roots: [1/2, 2, 3]
Sum: 11/2

Thought: The sum of the real roots is 11/2.
Action: {{"tool": "final_answer", "args": {{"answer": "11/2"}}}}

### Example 3 — Combinatorics (enumeration + early exit optimisation)
Problem: How many 4-digit numbers whose digits are each from 1 to 9 have digit \
sum exactly 10?

Thought: Enumerate all ordered 4-tuples (a,b,c,d) with each digit in 1-9 and \
a+b+c+d=10. Fix a,b,c and derive d to save one loop.
Action: {{"tool": "python_repl", "args": {{"code": "count = 0\\nfor a in range(1, 10):\\n    for b in range(1, 10):\\n        for c in range(1, 10):\\n            d = 10 - a - b - c\\n            if 1 <= d <= 9:\\n                count += 1\\nprint('Count:', count)"}}}}

Observation:
Count: 84

Thought: There are 84 such 4-digit numbers.
Action: {{"tool": "final_answer", "args": {{"answer": "84"}}}}

### Example 4 — Coordinate Geometry (sympy + exact radicals)
Problem: In triangle ABC, AB = 5, BC = 7, CA = 8. Find the area.

Thought: Use Heron's formula with exact sympy arithmetic to avoid rounding.
Action: {{"tool": "python_repl", "args": {{"code": "from sympy import sqrt, Rational, simplify\na, b, c = 7, 8, 5\ns = Rational(a + b + c, 2)\narea = sqrt(s * (s - a) * (s - b) * (s - c))\nprint('s =', s)\nprint('Area =', simplify(area))"}}}}

Observation:
s = 10
Area = 10*sqrt(3)

Thought: The area is 10√3.
Action: {{"tool": "final_answer", "args": {{"answer": "10*sqrt(3)"}}}}

---

Now solve the problem below. Use multiple python_repl calls if needed — \
variables persist between calls. Give an exact answer and verify before submitting.

Problem: {question}
"""

# Self-correction prompt

REFLECTION_PROMPT = """\
The previous action resulted in an error or unexpected outcome:
{error_info}

Please analyze what went wrong and try a different approach. \
Consider: Was the tool input malformed? Should you try a different search query? \
Is there a code bug to fix?
"""

STRUCTURED_REFLECTION_PROMPT = """\
The previous step entered an abnormal state and needs self-correction.

## Original Task
{task}

## Error Type
{error_type}

## Last Thought
{last_thought}

## Last Action
{last_action}

## Observation / Error
{observation}

## Recent Actions
{recent_actions}

## Available Tools
{tool_names}

## Recovery Rules
1. Diagnose the concrete cause before choosing the next action.
2. If this was a parse/JSON/tool-name error, output a valid Action JSON using one available tool.
3. If search returned no useful result, rewrite the query with fewer words, search an entity alias, or split the multi-hop question into an intermediate entity lookup.
4. If code failed, fix the code or write a smaller verification program. Prefer exact arithmetic.
5. If the same action was repeated, do not reuse the same Action JSON. Change the query/code, or answer from already gathered evidence if enough.
6. Do not mention this reflection prompt in the final answer.

Respond exactly in the normal format:

Thought: <brief diagnosis and recovery plan>
Action: {{"tool": "<tool_name>", "args": {{<arguments>}}}}
"""
