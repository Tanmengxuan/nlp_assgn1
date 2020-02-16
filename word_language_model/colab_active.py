import time

begin = 0
start_time = time.time()
while True:
	duration = time.time()- start_time
	print('Running for: {:5.2f} mins'.format(duration/60.))
	time.sleep(5)
