import socket
import lgpio
import time

HOST = "0.0.0.0"
PORT = 5000

LEFT_PIN = 5  
CENTER_PIN = 6
RIGHT_PIN = 26

# open gpio
h = lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(h, LEFT_PIN)
lgpio.gpio_claim_output(h, CENTER_PIN)
lgpio.gpio_claim_output(h, RIGHT_PIN)

def pulse(pin):
    lgpio.gpio_write(h, pin, 1)
    time.sleep(0.2)
    lgpio.gpio_write(h, pin, 0)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Waiting for signal...")

startTime = time.time()
currTime = time.time()

while currTime - startTime < 10:
    conn, addr = server.accept()
    data = conn.recv(1024).decode()

    print("Received:", data)

    if cmd == "LEFT":
        pulse(LEFT_PIN)

    elif cmd == "CENTER":
        pulse(CENTER_PIN)

    elif cmd == "RIGHT":
        pulse(RIGHT_PIN)

    conn.close()
    currTime = time.time()
