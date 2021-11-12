import mmcv

if __name__ == "__main__":
    video = mmcv.VideoReader("real01.mp4")
    video.cvt2frames('./real01', filename_tmpl='{:08d}.png')
