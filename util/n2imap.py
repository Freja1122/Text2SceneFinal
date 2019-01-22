import os
import json
from os.path import join
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

def get_n2i_map():
    # 映射单词的idx到具体的实例
    Idx2Instance = dict()
    lines = open("sources/AS/NounMI.txt", "r").readlines()
    for line in lines:
        line = line.split()
        Idx2Instance[int(line[0])] = int(line[1])

    # Noun2Instance:输入名词返回idx,在ClipArtIndices.png中可以找到idx对应的素材
    Noun2Instance = dict()
    lines = open("sources/AS/NounsMap.txt", 'r').readlines()
    for line in lines:
        line = line.split()
        try:
            Noun2Instance[line[2]] = Idx2Instance[int(line[0])]
        except:
            print(int(line[0]))
    return Noun2Instance


if __name__ == '__main__':
    Noun2Instance = get_n2i_map()
    print(Noun2Instance)


