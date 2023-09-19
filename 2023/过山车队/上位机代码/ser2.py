import serial as ser
import time

ser=ser.Serial('/dev/ttyTHS1',115200,timeout=0.5)
time.sleep(1)
ser.write(b'\x49')
ser.close()