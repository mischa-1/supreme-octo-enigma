import socket

HOST = "INSERT PI 0 IP"  # Pi Zero IP
PORT = 5000

def send(cmd):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send(cmd.encode())
    s.close()

send("LEFT")
