def ask_permission(request: str):
    result = input(request + " (yes/no)")
    if result.startswith("y"):
        return True
    return False