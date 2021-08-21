import mmcv
import os
from mmcv.video.processing import concat_video
from option import parse_args
from tqdm import tqdm
from tools import make_file_list


def main():
    """ Show your options """
    opt = parse_args()
    print(opt)

    assert os.path.exists(opt.video_path), f'{opt.video_path} is not exit!'

    """ Select a Video from file path """
    video_list = os.listdir(opt.video_path)
    video_list = [w for idx,w in enumerate(video_list) if w.endswith(('.avi','.AVI','.mp4','.MP4'))]

    assert len(video_list) > 0, f'your path:{opt.video_path} has no videos, at least put a clip in this folder!'
        
    print('Enter a filename which you want to resolve...')
    print(video_list)

    # video_name = input('Please enter your filename or index: ')
    video_name = '1'

    if not video_name.endswith(('.avi','.AVI','.mp4','.MP4')):
        if int(video_name) < 0 or int(video_name) >= len(video_list):
            print('please enter a valid name or index!')
            return 0
        idx = int(video_name)
        video_name = video_list[idx]

    """ 
        Now we deal with your video in mmcv tools
        videos --> frames: deaults 100 frames which we compute in one times  
    """

    video = os.path.join(opt.video_path, video_name)
    video = mmcv.VideoReader(video)

    # video_fps = video.fps

    print(f'There are {len(video)} frames in {video_name}')

    if len(video) < opt.limit_frames:  
        print('This video is too shot!')
        return

    if not os.path.exists(opt.tmp_dir):
        os.makedirs(opt.tmp_dir)

    if not os.path.exists(opt.tmp_dir_clip):
            os.makedirs(opt.tmp_dir_clip)

    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)


    clip_count = 0
    for idx, frame in tqdm(enumerate(video)):
        """ If the frames is not enough, we don't continue our work """
        if len(video) - idx < opt.limit_frames:
            os.system(f'rm {opt.tmp_dir}/*')
            break
        
        index = idx%100 # index: 0 ---> 100        
        file_name = str((idx%100)).zfill(8) + '.png'
        output = os.path.join(opt.tmp_dir, file_name)
        mmcv.imwrite(frame, output)

        if index == opt.limit_frames-1:
            opt.tmp_test_file_name = '/media/gky-u/DATA/ljj/VSR_TOOLS/test_file_list.txt' # Some models need a txt file
            make_file_list(opt.tmp_dir, opt.tmp_test_file_name)
            """ Put your frames in your model: STARnet or BasicVSR """
            
            os.system(f'python ../STARnet/eval.py --file_list {opt.tmp_test_file_name} --output {opt.tmp_dir}')

            mmcv.frames2video(opt.tmp_dir, opt.tmp_dir_clip + '/' + f'clip{clip_count}.mp4', fps=video.fps, filename_tmpl='{:08d}.png')
            clip_count += 1
            print(f'{idx} | {len(video)}')
            os.system(f'rm {opt.tmp_dir}/*')

    """ Super-resolution frames --> video """
    
    """ In this part, we delete the tmp* dirs and copy our clip in sr_videos """

    clips_list = [w for w in range(clip_count)]
    clips_list = [os.path.join(opt.tmp_dir_clip, f'clip{w}.mp4') for w in clips_list]


    mmcv.concat_video(clips_list, os.path.join(opt.results_dir, f'{video_name}'), log_level='quiet', vcodec='h264')
    os.system(f'rm -r {opt.tmp_dir}*')
    os.system(f'rm -r {opt.tmp_dir_clip}*')

# Start Test!
if __name__ == "__main__":
    main()
