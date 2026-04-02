import platform
import time
import lgpio

IS_PI = platform.system() == "Linux"

# Use BCM pin numbers (same numbers you used with GPIO.BCM)
LED_r = 34   # Vibration motor 1 
LED_R = 20

LED_m = 36  # Vibration motor 2
LED_M = 16

LED_l = 39    # Vibration motor 3
LED_L = 26

if IS_PI:
    import lgpio as _lgpio

    # On Raspberry Pi, gpiochip0 is typically the main controller.
    # 0 here means /dev/gpiochip0
    _chip_handle = _lgpio.gpiochip_open(0)

    # Claim pins as outputs, initial state LOW (0)
    _lgpio.gpio_claim_output(_chip_handle, LED_R, 0)
    _lgpio.gpio_claim_output(_chip_handle, LED_M, 0)
    _lgpio.gpio_claim_output(_chip_handle, LED_L, 0)

    def motors_off():
        _lgpio.gpio_write(_chip_handle, LED_R, 0)
        _lgpio.gpio_write(_chip_handle, LED_M, 0)
        _lgpio.gpio_write(_chip_handle, LED_L, 0)

    def motor_on(pin: int):
        _lgpio.gpio_write(_chip_handle, pin, 1)

    def motor_off_single(pin: int):
        _lgpio.gpio_write(_chip_handle, pin, 0)
else:
    def motors_off():
        pass

    def motor_on(pin: int):
        pass

 def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    startTime = time.time()

    motor_on(LED_R)
    motor_off_single(LED_M)
    motor_off_single(LED_L)

    time.sleep(3)

    motor_on(LED_M)
    motor_off_single(LED_R)
    motor_off_single(LED_L)

    time.sleep(3)

    motor_on(LED_L)
    motor_off_single(LED_M)
    motor_off_single(LED_R)

    time.sleep(3)

    motors_off()


if __name__ == "__main__":
    main()
