# -*- coding: utf-8 -*-
import socket


def main():
    """
    An example test for seed link
    """
    HOST = 'geofon.gfz-potsdam.de'    # The remote host
    PORT = 18000              # The remote port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('s.connect((', HOST, PORT, '))')
    s.connect((HOST, PORT))
    sendbytes = b'HELLO\r'
    print('Sent:', repr(sendbytes))
    s.send(sendbytes)
    data = s.recv(1024)
    print('Received:', repr(data))
    s.close()


if __name__ == "__main__":
    main()
