def load_dotenv(*args, **kwargs):
    pass

def dotenv_values(path: str = ".env", *_, **__) -> dict:
    """Minimal dotenv parser returning key=value pairs from given file.
    Lines starting with '#' are ignored. Values are not type-converted."""
    values: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                values[key.strip()] = val.strip()
    except FileNotFoundError:
        # Silently ignore missing file to match python-dotenv behaviour
        pass
    return values
