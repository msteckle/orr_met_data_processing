import imageio.v2 as imageio
base = '/home/6ru/Downloads'
filenames = [f'{base}/EDA_ Timeseries Charts 01.png', f'{base}/EDA_ Timeseries Charts 02.png', f'{base}/EDA_ Timeseries Charts 03.png',
             f'{base}/EDA_ Timeseries Charts 04.png', f'{base}/EDA_ Timeseries Charts 05.png', f'{base}/EDA_ Timeseries Charts 06.png',
             f'{base}/EDA_ Timeseries Charts 07.png', f'{base}/EDA_ Timeseries Charts 08.png', f'{base}/EDA_ Timeseries Charts 09.png',
             f'{base}/EDA_ Timeseries Charts 10.png', f'{base}/EDA_ Timeseries Charts 11.png', f'{base}/EDA_ Timeseries Charts 12.png']
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('/home/6ru/Desktop/tempc_months_ts.gif', images, fps=2, loop=0)