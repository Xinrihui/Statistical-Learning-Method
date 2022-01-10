from multiprocessing import Pool
import os, time, random

import numpy as np


def task(i):

    print('Run task pid: %s...' % ( os.getpid())) # 每次 在进程池中拿一个 进程

    for j in range(i):
        j += 1

    a=np.sqrt(np.log(i ** 2 + 1))


    return a



def multiprocess(n,nthread=8):
    """

    多进程 并发

    :return:
    """

    print('Parent process %s.' % os.getpid())

    p = Pool(nthread) # 进程池, 和系统申请 nthread 个进程

    cache=[0]*n

    for i in range(n):

        cache[i]=p.apply_async(task, args=(i,)) # apply_async 异步调用

    print('Waiting for all subprocesses done...')
    p.close()

    print('All subprocesses done.')

    p.join()

    for i in range(n):
        cache[i] = cache[i].get(1) # 得到结果

    return cache

def singleprocess(n):

    cache=[0]*n

    for i in range( n):

        for j in range(i): # 时间复杂度: O(n^2)
            j+=1

        cache[i] =  np.sqrt(np.log(i ** 2 + 1))

    return cache

if __name__=='__main__':

    n=20000

    # start = time.time()
    #
    # res=singleprocess(n)
    #
    # end = time.time()
    # print(res[:20])
    # print('single processs runs %0.2f seconds.' % ((end - start))) # 12.73s

    start = time.time()

    res=multiprocess(n,nthread=4)

    end = time.time()

    print(res[:20])

    print('multi processs runs %0.2f seconds.' % ((end - start)))

    # nthread=2  time: 11.57
    # nthread=4  time: 6.75
    # nthread=8  time: 6.01
    # nthread=16  time: 6.78