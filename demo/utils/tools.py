""" Make a file list (txt) for a model"""
import os
import pandas as pd

def make_file_list(path, save_file):
    list = os.listdir(path)
    list.sort()

    file_list = [path + '/'+ img.rstrip('.png') for img in list]

    save_file = pd.DataFrame(file_list)
    # file_list = []
    # for clip in list:
    #     clip_dir = os.path.join(path, clip)
    #     img_list = os.listdir(clip_dir)
    #     img_list.sort()

    #     tmp = [clip+ '/' +w for w in img_list]
    #     file_list.extend(tmp)
    
    print(file_list)

if __name__ == "__main__":
    path = './test'
    save_file = "test_file_list.txt"
    make_file_list(path, save_file)



    
