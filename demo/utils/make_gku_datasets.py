import os
import mmcv

def make_gky_datasets(input_dir, rate=0.8, sr=True):
    list = os.listdir(input_dir)
    list.sort()
    print(f'Totol Clips: {len(list)}')
    
    index = int(len(list) * rate)
    train_clips, test_clips = list[:index], list[index:]
    print(f'Training clips:{len(train_clips)} | Testing clips:{len(test_clips)}')
    
    save_dir = os.path.dirname(input_dir)
    basename = os.path.basename(input_dir)

    save_dir = os.path.join(save_dir, basename+'_train_frames') # Generate the training video frames
    print('save_train_dir: ',save_dir)
    
    for idx, clip in enumerate(train_clips):
        clip_folder = '{:03d}'.format(idx)
        tmp_clip_save_dir = os.path.join(save_dir, clip_folder)
        video = os.path.join(input_dir, train_clips[idx])
        video = mmcv.VideoReader(video)
        video.cvt2frames(tmp_clip_save_dir, filename_tmpl='{:08d}.png')
        
        print(f'Processing Training clips: {idx}|{len(train_clips)}')
    
    save_dir = os.path.join(save_dir, basename+'_test_frames') # Generate the testing video frames
    print('save_test_dir: ',save_dir)

    for idx, clip in enumerate(test_clips):
        clip_folder = '{:03d}'.format(idx)
        tmp_clip_save_dir = os.path.join(save_dir, clip_folder)
        video = os.path.join(input_dir, test_clips[idx])
        video = mmcv.VideoReader(video)
        video.cvt2frames(tmp_clip_save_dir, filename_tmpl='{:08d}.png')
        
        print(f'Processing Testing clips: {idx}|{len(train_clips)}')
        
if __name__ == '__main__':
    # input_dir = '/media/gky-u/DATA/ljj/DATA/gky_data/SDR_4K'
    input_dir = '/media/gky-u/DATA/ljj/DATA/gky_data/SDR_540p'
    make_gky_datasets(input_dir)
    
