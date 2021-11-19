import json, sys, os
import matplotlib.pylab as plt

def plot_psnr_ssim(data_path, defualt_option = 'psnr', max_iter = 1000000):
    iters, psnr_list, ssim_list = [], [], []
    plt.ion()
    plt.rcParams['font.sans-serif']=['SimHei']   # 防止中文标签乱码，还有通过导入字体文件的方法
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 1.5   #设置曲线线条宽度
    plt.xlabel('Iterations')
    
    if defualt_option == 'psnr':
        plt.ylabel('PSNR(dB)')
    else: plt.ylabel('SSIM')

    with open(data_path, 'r') as f:
        content = f.readlines() 

        for line in content:
            data = json.loads(line)  
            try:    
                iter, psnr, ssim = data['iter'], data['PSNR'], data['SSIM']

                iters.append(iter)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                
                if defualt_option == 'psnr':
                    plt.plot(iters, psnr_list, 'g-')
                    plt.pause(1)  # 暂停两秒
                else:
                    plt.plot(iters, ssim_list, 'g-')
                    plt.pause(1)  # 暂停两秒
            
            except: 
                print('start !')
        
        
        plt.savefig('{}_curve.png'.format(defualt_option), dpi=300)  # 设置保存图片的分辨率
        plt.ioff()       # 关闭画图的窗口，即关闭交互模式
        plt.show()       # 显示图片，防止闪退


if __name__ == "__main__":
    # read_json(sys.argv[1])
    log_list = os.listdir(os.getcwd())
    for idx, file in enumerate(log_list):
        if 'log' in os.path.basename(file):
            print("{}. {}".format(idx, file))
    
    input_number = input("Please choose a log file: ")
    input_number = int(input_number)

    path = log_list[input_number]
    # plot_psnr_ssim('./20210806_150047.log.json')
    plot_psnr_ssim(path, 'psnr')
    plot_psnr_ssim(path, 'ssim')
    
