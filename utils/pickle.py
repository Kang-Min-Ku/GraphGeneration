import pickle
import json

def load_pickle(io):
    with open(io, "rb") as file:
        content = pickle.load(file)
    return content

def save_pickle(content, io):
    with open(io, "wb") as file:
        pickle.dump(content, file)
    return

def pickle_to_json(pkl_file, json_file):
    content = load_pickle(pkl_file)
    with open(json_file, "w+") as file:
        json.dump(content, file)
    return