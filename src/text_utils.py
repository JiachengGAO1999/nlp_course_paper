import re


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text):
    """Lightweight tokenizer proxy for budget checks before model-token auditing."""
    return len(TOKEN_RE.findall(text or ""))


def render_dialogue(messages):
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n\n".join(lines)


def render_question(question, options):
    option_lines = [f"{key}. {value}" for key, value in options.items()]
    return question + "\n\n" + "\n".join(option_lines)


def approximate_fit(items, budget, formatter):
    selected = []
    for item in reversed(items):
        candidate = [item] + selected
        text = formatter(candidate)
        if count_tokens(text) <= budget or not selected:
            selected = candidate
        else:
            break
    return selected, formatter(selected)
