import time
import lgpio
import argparse

parse = argparse.ArgumentParser(description= "Date for this program")
parse.add_argument("--tim", action= "store", type=int, default = 5, help = "time for loop")
args = parse.parse_args()

chip = lgpio.gpiochip_open(0)

TRIG = 17
ECHO = 27

#lgpio.gpiochip_close(chip)

lgpio.gpio_claim_output(chip, TRIG)
lgpio.gpio_write(chip, TRIG, 0) 

lgpio.gpio_claim_input(chip, ECHO)

def convert(time):
  return (time *34300)/2

startTime = time.time()

while(time.time() < startTime + args.tim):
  #time0 = time.time()
  lgpio.gpio_write(chip, TRIG, 1)
  time.sleep(0.00001)
  lgpio.gpio_write(chip, TRIG, 0)
  while(not lgpio.gpio_read(chip, ECHO)):
    time0 = time.time()
  while(lgpio.gpio_read(chip,ECHO)):
    time1 = time.time()
  distance = convert(time1-time0)
  print (f'Distance Away: {distance:.2f} cm')
  time.sleep(0.1)

lgpio.gpiochip_close(chip)

