import mmcv

if __name__ == "__main__":
    video = mmcv.VideoReader("flowers.mp4")
    video.cvt2frames('./frame', filename_tmpl='{:08d}.png')
