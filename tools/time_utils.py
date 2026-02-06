import time

start_time = time.time()
current_time = time.time()

def show_time(start_time,current_time):
    start_time = time.localtime(start_time)
    current_time = time.localtime(current_time)
    tm_hour = current_time.tm_hour - start_time.tm_hour 
    tm_min = current_time.tm_min - start_time.tm_min  
    tm_sec = current_time.tm_sec - start_time.tm_sec 
    time_string = f"{tm_hour}时{tm_min}分{tm_sec}秒"
    return time_string

print(show_time(start_time,current_time))