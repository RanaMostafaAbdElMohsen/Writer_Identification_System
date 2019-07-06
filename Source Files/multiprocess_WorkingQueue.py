from multiprocessing import Queue,Process,freeze_support
import time 
import random

def worker(num,queue):
    """thread worker function"""
    for i in range(100):
        time.sleep(random.uniform(0.500,0.100))
        print ('Worker: %s %s' % (num,i))
        queue.put(i)
    return


if __name__ == '__main__':
    threads=[]
    freeze_support()
    out_queue1 = Queue()
    for i in range(5):
        t = Process(target=worker, args=(i,out_queue1,))
        t.start()
        threads.append(t)
    
    for i in threads:
        i.join()
    print(out_queue1)