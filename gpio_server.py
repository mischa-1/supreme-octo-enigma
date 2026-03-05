import socket
import lgpio
import time

HOST = "0.0.0.0"
PORT = 5000

GPIO_PIN = 21

# open gpio
h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, GPIO_PIN)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Waiting for signal...")

while True:
    conn, addr = server.accept()
    data = conn.recv(1024).decode()

    print("Received:", data)

    if data == "ON":
        lgpio.gpio_write(h, GPIO_PIN, 1)

    elif data == "OFF":
        lgpio.gpio_write(h, GPIO_PIN, 0)

    elif data == "PULSE":
        lgpio.gpio_write(h, GPIO_PIN, 1)
        time.sleep(0.2)
        lgpio.gpio_write(h, GPIO_PIN, 0)

    conn.close()
