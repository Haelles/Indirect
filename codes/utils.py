import json
import os


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(file, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(file, f, ensure_ascii=False, indent=4)