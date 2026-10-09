def print_green(text): #user text
    print(f"\033[92m{text}\033[0m", end="", flush=True)

def print_yellow(text): #llm text
    print(f"\033[93m{text}\033[0m", end="", flush=True)

def print_red(text): #commands
    print(f"\033[91m{text}\033[0m", end="", flush=True)

def print_blue(text): #no use yet
    print(f"\033[94m{text}\033[0m", end="", flush=True)