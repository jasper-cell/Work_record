###客户端client.py
import socket
import os
import sys
import struct
import cv2
import base64
import json
import numpy as np
from threading import Thread
import pickle
from common_objs import tcp_package
import struct


def receive():
    while True:
        in_data = bytes()
        data = client.recv(1024)
        print("client received ...")

        while data:
            in_data += data
            data = client.recv(1024)
            if(len(data) < 1024):
                break

        print(len(data))
        in_data += data
        # data_recv = pickle.loads(in_data)
        data_recv = struct.unpack(in_data)
        print("obj_info: ", data_recv.obj_info)
        cv2.imshow("image_client", data_recv.mask_data)
        cv2.waitKey(0)


def send():
    while True:
        image_name = input(">> ")
        print("image_name", image_name)
        if os.path.exists(image_name):
            image = cv2.imread(str(image_name))
            data_original = tcp_package(width=image.shape[0], height=image.shape[1], obj=None,
                        src_jpg_length=len(image), mask_jpg_length=None, result_jpg_length=None,
                        src_data=image, mask_data=None, result_data=None)
            # data_send = pickle.dumps(data_original)
            data_send = struct.pack(image)
            client.send(data_send)
        else:
            continue

if __name__ == '__main__':
    client = socket.socket()
    client.connect(('127.0.0.1', 20010))

    t3 = Thread(target=send, name= 3)
    t4 = Thread(target=receive, name=4)
    t3.start()
    t4.start()

