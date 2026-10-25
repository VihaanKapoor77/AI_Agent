"""Readable terminal output, with optional Rich Markdown rendering."""
import json
try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console(markup=False)
except ImportError:
    console = None


def show(title, content):
    if isinstance(content, dict):
        lines = []
        for key, value in content.items():
            label = key.replace('_', ' ').capitalize()
            if isinstance(value, list):
                lines.append(label + ':')
                lines.extend('  - ' + (json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)) for item in value)
            else:
                lines.append(f'{label}: {value}')
        text = '\n'.join(lines)
    elif isinstance(content, list):
        text = '\n'.join('- ' + str(item) for item in content)
    else:
        text = str(content)
    if console:
        console.rule(title)
        if title in {'Assistant', 'Update'}:
            console.print(Markdown(text))
        else:
            console.print(text, highlight=False)
    else:
        print(f'\n{title}\n' + '-' * min(len(title), 60))
        print(text, flush=True)
