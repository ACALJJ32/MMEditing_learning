# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import shutil
import sys

import cv2
import lmdb
import mmcv


def make_lmdb(mode, data_path, lmdb_path, batch=5000, compress_level=1):
    """Create lmdb for the REDS dataset.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    Args:
        mode (str): REDS dataset mode. Choices: ['train_sharp', 'train_blur',
            'train_blur_comp', 'train_sharp_bicubic', 'train_blur_bicubic'].
            They are used to identify different reds dataset for different
            tasks. Specifically:
            'train_sharp': GT frames;
            'train_blur': Blur frames for deblur task.
            'train_blur_comp': Blur and compressed frames for deblur and
                compression task.
            'train_sharp_bicubic': Bicubic downsampled sharp frames for SR
                task.
            'train_blur_bicubic': Bicubic downsampled blur frames for SR task.
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
    """

    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    if mode in ['train_sharp', 'train_blur', 'train_blur_comp']:
        h_dst, w_dst = 720, 1280
    else:
        h_dst, w_dst = 180, 320

    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(data_path, suffix='png', recursive=True)))
    keys = []
    for img_path in img_path_list:
        parts = img_path.split('/')
        folder = parts[-2]
        img_name = parts[-1].split('.png')[0]
        keys.append(folder + '_' + img_name)  # example: 000_00000000

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_path_list[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_path_list)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    pbar = mmcv.ProgressBar(len(img_path_list))
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update()
        key_byte = key.encode('ascii')
        img = mmcv.imread(osp.join(data_path, path), flag='unchanged')
        h, w, c = img.shape
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        assert h == h_dst and w == w_dst and c == 3, (
            f'Wrong shape ({h, w}), should be ({h_dst, w_dst}).')
        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess REDS datasets',
        epilog='You can first download REDS datasets using the script from:'
        'https://gist.github.com/SeungjunNah/b10d369b92840cb8dd2118dd4f41d643')
    parser.add_argument('--root-path', type=str, help='root path for REDS')
    parser.add_argument(
        '--make-lmdb', action='store_true', help='create lmdb files')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    root_path = args.root_path

    train_folders = [
        'train_sharp', 'train_sharp_BD', 'train_sharp_bicubic'
    ]

    

    # create lmdb file
    if args.make_lmdb:
        for train_folder in train_folders:
            lmdb_path = osp.join(root_path, train_folder + '.lmdb')
            data_path = osp.join(root_path, train_folder)
            if 'bicubic' in train_folder:
                data_path = osp.join(data_path, 'X4')
            make_lmdb(
                mode=train_folder, data_path=data_path, lmdb_path=lmdb_path)
