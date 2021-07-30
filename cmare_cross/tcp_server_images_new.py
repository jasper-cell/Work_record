###服务器端server.py
import socket
from size_caculate_images_tcp import integerate
import struct
import numpy as np
import cv2
from sys import getsizeof
from io import BytesIO
from PIL import Image
import traceback

if __name__ == '__main__':
    image_info = None
    obj_info = None
    mask_info = None

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # s只负责监听
    s.bind(('127.0.0.1', 6615))
    s.listen(5)

    print("Wait for Connection.....................")
    while True:
        sock, addr = s.accept()  # addr是一个元组(ip,port), 使用的是新的套接字
        print("Accept connection from {0}".format(addr))  # 查看发送端的ip和端口
        while True:
            try:
                in_data = bytes()
                header_data = sock.recv(56)  # 会阻塞再这里直到有数据流过来
                struct_data = struct.unpack_from("<QQiiiffffiii", header_data, 0)
                print("header length: ", len(header_data))
                print("===========================================")
                rest_len=struct_data[9]
                while rest_len>0:
                    data = sock.recv(rest_len)
                    in_data += data
                    rest_len-=len(data)

                print("===========================================")

                print("in_data length: ", getsizeof(in_data))
                '''
                二进制解析为numpy
                '''
                img = Image.open(BytesIO(in_data))
                imgm=np.asarray(img)
                img = cv2.cvtColor(imgm, cv2.COLOR_RGB2BGR)
                print("acquire...")

                '''
                进行图像处理
                '''
                print("processing image ...")
                res, mask_ret = integerate(img)
                print("processing compeleted !!!")

                '''
                解析对应的检测到的物体信息
                '''

                obj_info = res[1]
                obj_width = obj_info[1]['width']
                obj_height = obj_info[1]['height']
                obj_area = obj_info[1]['area']
                obj_cir = obj_info[1]['arclength']

                result = res[0]  # 结果图像
                mask = mask_ret  # mask图像
                src_jpg_pic_length = 0  # 输入图像的长度, 返回时不需要输入图像的长度

                '''
                将产生的结果转换为二进制文件
                '''
                mask_PIL = Image.fromarray(mask)
                result_PIL = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

                mask_binary = BytesIO()
                result_binary = BytesIO()

                mask_PIL.save(mask_binary, format='jpeg')  # 将二进制文件存储到对应的变量
                result_PIL.save(result_binary, format='jpeg')

                mask_jpg_length = len(mask_binary.getvalue())  # mask图像的length
                result_jpg_length = len(result_binary.getvalue())  # result图像的length

                print("mask_jpg_length", mask_jpg_length)
                print("result_jpg_length", result_jpg_length)

                '''
                修改对应的结构体的成员
                '''
                struct_data = list(struct_data)

                struct_data[2] = 0x2   # 返回结果类型
                struct_data[5] = float(obj_width)  # 宽
                struct_data[6] = float(obj_height)  # 高
                struct_data[7] = float(obj_area)  # 面积
                struct_data[8] = float(obj_cir)  # 弧长
                struct_data[9] = src_jpg_pic_length  # 为 0
                struct_data[10] = mask_jpg_length  # mask length
                struct_data[11] = result_jpg_length  # result length

                struct_data = tuple(struct_data)

                unstruct_data = struct.pack("<QQiiiffffiii", *struct_data)
                print("unstruct_data len: ", len(unstruct_data))

                '''
                合并发送数据
                '''
                send_data = unstruct_data + mask_binary.getvalue() + result_binary.getvalue()
                sock.send(send_data)
            except Exception as e:
                traceback.print_exc()
                print (e)
                break
        # print("socket close")
        sock.close()
