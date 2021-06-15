import cv2
import glob

img_root = 'D:\\Work\\doc\\nex-code-main\\crest'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 30    #保存视频的FPS，可以适当调整
size=(1008, 756)
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('D:/3.avi', fourcc, fps, size)#最后一个是保存图片的尺寸

image_paths = glob.glob(img_root + "/*.png", recursive=True)

for i in image_paths:
    frame = cv2.imread(i)
    videoWriter.write(frame)

videoWriter.release()
