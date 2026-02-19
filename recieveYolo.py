import time
import lgpio

LED_L = 23
LED_M = 24
LED_R = 25

chip_handle = lgpio.gpiochip_open(0)

lgpio.gpio_claim_output(_chip_handle, LED_R, 0)
lgpio.gpio_claim_output(_chip_handle, LED_M, 0)
lgpio.gpio_claim_output(_chip_handle, LED_L, 0)

def leds_off():
  lgpio.gpio_write(chip_handle, LED_R, 0)
  lgpio.gpio_write(chip_handle, LED_M, 0)
  lgpio.gpio_write(chip_handle, LED_L, 0)

def main():
  try:
    while True:
      # recieve and turn on and off led
  except KeyboardInterrupt:
        pass
    
  finally:
    leds_off()
    lgpio.gpiochip_close(chip_handle)
