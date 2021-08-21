import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='super-resolution demo')
    parser.add_argument('--video_path', type=str, default='./video', help='Your Video path')
    parser.add_argument('--output', type=str, default='./sr_videos', help='Put your super resolution video in this place')
    parser.add_argument('--limit_frames', type=int, default=100, help='input frames')

    parser.add_argument('--gpus', type=int, default=1, help='GPUS')
    parser.add_argument('--tmp_dir', type=str, default='./000', help='Put some imgs in this dir')
    parser.add_argument('--tmp_dir_clip', type=str, default='./tmp_dir_clip', help='Put some clips in this dir')
    parser.add_argument('--user', type=str, default='user',help='define your user id')
    parser.add_argument('--results_dir', type=str, default='./sr_videos',help='put your final videos in this place')
    
    parser.add_argument('--model_type', type=str, default='STARnet',help='STARnet or mmediting')
    parser.add_argument('--tmp_test_file_name', type=str, default='/media/gky-u/DATA/ljj/VSR_TOOLS/test_file_list.txt',\
        help='Image File List')

    return parser.parse_args()

