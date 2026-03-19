import socket
import time

HOST = "172.27.61.223 "  # Pi Zero IP
PORT = 5000

def send(cmd):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send(cmd.encode())
    s.close()

for i in range(5)
    send("LEFT")
    time.sleep(0.4)
