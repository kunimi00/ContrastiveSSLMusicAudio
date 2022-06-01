"""
2021/06/01 Jeong Choi
데이터 저장 및 불러오기 용도 함수

"""

import pickle
import json

def save_obj_to_pickle(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


## TXT format
def save_list_to_txt(obj, path):
    with open(path, 'w', encoding="utf-8") as f:
        for item in obj:
            f.write("%s\n" % item)

def load_list_from_txt(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    return lines


def save_dict_to_json(obj, path):
    with open(path, 'w') as outfile:
        json.dump(obj, outfile, indent=4)


def load_dict_from_json(path):
    f = open(path, encoding="latin-1")
    js = f.read()
    f.close()
    return json.loads(js)


