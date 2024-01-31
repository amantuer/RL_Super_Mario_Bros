import time
import datetime

def current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

class ExecutionTimer:
    def __init__(self):
        self.start_time = 0
        self.recorded_times = []

    def begin(self):
        self.start_time = time.time()

    def log_duration(self, message=''):
        duration = time.time() - self.start_time
        print(f"{message} Time taken: {duration}")

    def current_duration(self):
        return time.time() - self.start_time
    
    def save_duration(self):
        self.recorded_times.append(self.current_duration())

    def average_time(self):
        return sum(self.recorded_times) / len(self.recorded_times) if self.recorded_times else 0
