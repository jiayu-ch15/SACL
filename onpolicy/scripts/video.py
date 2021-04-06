
import imageio
import os
import os.path as osp


def img2gif(img_dir, gif_path, duration):
    """

    :param img_dir: 包含图片的文件夹
    :param gif_path: 输出的gif的路径
    :param duration: 每张图片切换的时间间隔，与fps的关系：duration = 1 / fps
    :return:
    """
    frames = []
    for idx in range(640):
        img = osp.join(img_dir, 'step-0{:0>3d}.png'.format(idx)  )
        frames.append(imageio.imread(img))

    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)
    print('Finish changing!')


if __name__ == '__main__':
    
    img_dir = '/home/yuchao19/project/onpolicy/onpolicy/scripts/results/Habitat/mappo/debug/run132/gifs/Adrian/episode_1/all/' 
   
         
    par_dir = osp.dirname(img_dir)
    gif_path = osp.join(par_dir, 'output1.gif')
    img2gif(img_dir=img_dir, gif_path=gif_path, duration=0.1)