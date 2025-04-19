import serial as pyserial
import time

class SerialController:
    def __init__(self, port='COM5', baudrate=115200, timeout=1):
        self.ser = pyserial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # Allow time for serial connection to establish
        
    def start_trigger(self):
        self.ser.write(b'1')
        print("Trigger started")
        
    def stop_trigger(self):
        self.ser.write(b'0')
        print("Trigger stopped")
        
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed") 