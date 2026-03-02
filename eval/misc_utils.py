import re
from typing import List, Dict, Optional

# Match <|im_start|>role ... until next <|im_start|> or end-of-string
SEG_RE = re.compile(
    r"<\|im_start\|>\s*(system|user|assistant)\s*(.*?)(?=(?:<\|im_start\|>)|$)",
    re.DOTALL
)

REASON_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
ANSWER_RE  = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def maybe_unescape_newlines(raw: str) -> str:
    # If the text contains lots of literal "\n" and few/no real newlines, unescape them.
    if "\\n" in raw and raw.count("\n") < 2:
        raw = raw.replace("\\n", "\n")
    return raw

def parse_qwen_chat(raw: str) -> List[Dict[str, str]]:
    raw = maybe_unescape_newlines(raw)
    msgs = []
    for role, content in SEG_RE.findall(raw):
        content = content.replace("<|im_end|>", "").strip()
        msgs.append({"role": role, "content": content})
    return msgs

def extract_tag_block(content: str, regex: re.Pattern) -> Optional[str]:
    m = regex.search(content)
    return m.group(1).strip() if m else None

_DECIMAL = "<DECIMAL_DOT>"
_LISTDOT = "<LIST_DOT>"
def split_reasoning_into_steps(reasoning: str):
    # Normalize whitespace (treat newlines as spaces)
    text = re.sub(r"\s+", " ", reasoning.strip())

    # Protect decimal points between digits: 1.23, 0.5, 10.0, etc.
    text = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL, text)

    # Protect numeric list markers: " 1. " / " 2. " (but not decimals)
    # This turns "1." into "1<LIST_DOT>" when it looks like an enumerator.
    text = re.sub(r"(?:(?<=^)|(?<=\s))(\d{1,3})\.(?=\s)", r"\1" + _LISTDOT, text)

    # Optional: also treat "1)" as a list marker (uncomment if you want)
    text = re.sub(r"(?:(?<=^)|(?<=\s))(\d{1,3})\)(?=\s)", r"\1<LIST_RPAREN>", text)

    # Split points:
    #  - sentence end punctuation . ! ? followed by space/end
    #  - OR before a list marker " <num><LIST_DOT> "
    split_re = re.compile(
        r"""
        .*?(
            (?:[.!?](?=\s|$))     # sentence end
          | (?=\s+\d{1,3}""" + re.escape(_LISTDOT) + r""")  # before list item
          | $                     # end
        )
        """,
        re.VERBOSE,
    )

    parts = [m.group(0).strip() for m in split_re.finditer(text)]
    parts = [p for p in parts if p]

    # Restore protected tokens
    parts = [p.replace(_LISTDOT, ".").replace(_DECIMAL, ".") for p in parts]

    # Post-process: if you have "must:" as its own chunk, you can merge it with next
    merged = []
    for p in parts:
        if merged and re.search(r":\s*$", merged[-1]) and re.match(r"^\d{1,3}\.\s", p):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)

    return merged

def extract_question_and_steps(raw: str):
    msgs = parse_qwen_chat(raw)

    user_msgs = [m for m in msgs if m["role"] == "user"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]

    last_user = user_msgs[-1]["content"] if user_msgs else ""
    last_asst = asst_msgs[-1]["content"] if asst_msgs else ""

    # these tags are not really needed for evaluating 
    # how correct a chain of thought is
    reasoning = last_asst.replace("<reasoning>", "")
    reasoning = reasoning.replace("</reasoning>", "")
    reasoning = reasoning.replace("<answer>", "")
    reasoning = reasoning.replace("</answer>", "")
    reasoning = reasoning.replace("\\boxed{", "")

    # remove start and end new lines, if present
    while reasoning.startswith("\n"):
        reasoning = reasoning[1:]
    while reasoning.endswith("\n"):
        reasoning = reasoning[:-1]

    steps = split_reasoning_into_steps(reasoning) if reasoning else []
    return {"question": last_user, "reasoning": reasoning, "steps": steps}
        
def extract_question_and_steps_coding(raw: str):
    """Extract question, reasoning, and code from Qwen-style chat log for coding tasks.
    Currently works for mbpp only and for the specific prompts used."""
    msgs = parse_qwen_chat(raw)

    user_msgs = [m for m in msgs if m["role"] == "user"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]

    last_user = user_msgs[-1]["content"] if user_msgs else ""
    last_asst = asst_msgs[-1]["content"] if asst_msgs else ""

    # these tags are not really needed for evaluating 
    # how correct a chain of thought is
    if '<reasoning>' in last_asst and '```python' in last_asst:
        reasoning = last_asst.split('<reasoning>')[1].split('```python')[0]
        code = '```python' + last_asst.split('<reasoning>')[1].split('```python')[1].split('```')[0]
    else:
        reasoning = ''
        code = last_asst.strip('<reasoning>').strip()

    # remove start and end new lines, if present
    while reasoning.startswith("\n"):
        reasoning = reasoning[1:]
    while reasoning.endswith("\n") or reasoning.endswith("<>"):
        if reasoning.endswith("<>"):
            reasoning = reasoning[:-2]
        else:
            reasoning = reasoning[:-1]

    steps = split_reasoning_into_steps(reasoning) if reasoning else []
    return {"question": last_user, "reasoning": reasoning, "code": code, "steps": steps}

def extract_hint_lines(diffusion_text: str, max_lines: int = 10) -> list[str]:
    # Prefer <comments>, fallback to <reasoning>
    m = re.search(r"<comments>\s*(.*?)(?:<tests>|$)", diffusion_text, flags=re.S | re.I)
    if not m:
        m = re.search(r"<reasoning>\s*(.*?)(?:<implementation>|$)", diffusion_text, flags=re.S | re.I)
    content = (m.group(1) if m else diffusion_text).strip()

    # Remove fenced code blocks entirely
    content = re.sub(r"```.*?```", "", content, flags=re.S)

    # Split into candidate lines (handle numbered lists + paragraphs)
    raw = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[\-\*\d]+\s*[\.\)]\s*", "", line)  # strip bullets/1)/1.
        raw.append(" ".join(line.split()))

    if len(raw) <= 2:  # likely one paragraph -> split sentences
        raw = re.split(r"(?<=[.!?])\s+", content)

    hints = []
    for line in raw:
        line = " ".join(line.split())
        if not line:
            continue
        # Drop obvious code-ish lines
        if any(tok in line for tok in ("```", "def ", "class ", "import ")):
            continue
        hints.append(line)
        if len(hints) >= max_lines:
            break
    return hints