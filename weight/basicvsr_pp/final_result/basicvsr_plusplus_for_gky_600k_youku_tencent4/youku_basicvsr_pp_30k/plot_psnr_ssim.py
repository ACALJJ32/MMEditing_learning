import json, sys, os
import matplotlib.pylab as plt
import numpy as np

def plot_psnr_ssim(data_path):
    iters, psnr_list, ssim_list = [], [], []

    with open(data_path, 'r') as f:
        content = f.readlines() 

        for line in content:
            data = json.loads(line)  
            try:    
                iter, psnr = data['iter'], data['PSNR']

                iters.append(iter)
                psnr_list.append(psnr)
             
            except: 
                error_ = ''
        
    return iters, psnr_list
        
def plot_curve(data, iters, ylabel='PSNR'):
    plt.ion()
    plt.rcParams['font.sans-serif']=['SimHei']   # 防止中文标签乱码，还有通过导入字体文件的方法
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 1.5   #设置曲线线条宽度
    plt.xlabel('Iterations')
    
    if ylabel == 'PSNR':
        plt.ylabel('PSNR(dB)')

    if ylabel == 'PSNR':
        plt.plot(iters, psnr_list, 'g-')
        plt.pause(1)  # 暂停两秒
          
    plt.savefig('{}_curve.png'.format(ylabel), dpi=300)  # 设置保存图片的分辨率
    plt.ioff()       # 关闭画图的窗口，即关闭交互模式
    plt.show()       # 显示图片，防止闪退


if __name__ == "__main__":
    # read_json(sys.argv[1])
    log_list = os.listdir(os.getcwd())
    for idx, file in enumerate(log_list):
        if 'log' in os.path.basename(file):
            print("{}. {}".format(idx, file))
    
    files = input("Please enter number of your files: ")
    file_num = int(files)

    iter_list, psnr_list, ssim_list = [], [], []

    for i in range(file_num):
        input_number = input("Please choose a log file: ")
        input_number = int(input_number)

        path = log_list[input_number]
        iter_temp, psnr_temp = plot_psnr_ssim(path)

        psnr_list.extend(psnr_temp)
        iter_list.extend(iter_temp)
    
    plot_curve(psnr_list, iter_list, 'PSNR')
    print("max: ", max(psnr_list))
    print("max index: ", iter_list[np.argmax(psnr_list)])
    

    
