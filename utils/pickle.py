import pickle

def load_pickle(io):
    with open(io, "rb") as file:
        content = pickle.load(file)
    return content

def save_pickle(content, io):
    with open(io, "wb") as file:
        pickle.dump(content, file)
    return