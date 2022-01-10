from multiprocessing import Pool
import os, time

import numpy as np

from multiprocessing.managers import SharedMemoryManager


class Node:
    def __init__(self,key,value):
        self.key=key
        self.value = value

class Solution:

    def task(self,cache,shm_a,i):

        print('Run task pid: {} ;  cache_name is {}' .format ( os.getpid(),cache.shm.name)) # 每次 在进程池中拿一个 进程, nthread=4则进程号只有4个
                                                                                   # 一共4个进程 , 当前task 结束后给 进程别的task 复用
        # 子进程必须加 try-catch 否则出错了都不知道
        try:

            for j in range(i):
                j += 1

            cache[i]= float(np.sqrt(np.log(i ** 2 + 1))) # 必须转换为 float

            print('shm_a id in sub process: {} '.format(id(shm_a)))

            # shm_a.buf[i]= Node(i,10) # Error: memoryview: invalid type for format 'B'
            # shm_a.buf[i] = float(0.02) # Error: memoryview: invalid type for format 'B'

            shm_a.buf[i] = int(i) #


            # self.global_array[i]=i
            # print(self.global_array)

            # print('array id in sub process: {} '.format(id(self.global_array))) # 每个进程 复制了一个 global_array, 和主进程不是一个array,
            #                                                                     # nthread=4 , 因为进程复用, 内存空间中一共有 4个 global_array
            # print('string id in sub process: {} '.format(id(self.global_string))) # 每个 task 都有一个 global_string
            #
            # print('int id in sub process: {} '.format(id(self.global_int))) # 和主进程 复用一个 global_int

        except Exception as e:
            print('Error:', e)
        finally:
            print('finally...')

    def multiprocess(self,n,nthread=8):
        """

        多进程 并发

        :return:
        """

        print('Parent process %s.' % os.getpid())

        p = Pool(nthread) # 进程池, 和系统申请 nthread 个进程

        smm = SharedMemoryManager() #TODO: pyrhon3.8+ 才有
        smm.start()  # Start the process that manages the shared memory blocks

        cache_list = smm.ShareableList([0]*n)
        # 限制了可被存储在其中的值只能是 int, float, bool, str （每条数据小于10M）, bytes （每条数据小于10M）以及 None 这些内置类型。
        # 它另一个显著区别于内置 list 类型的地方在于它的长度无法修改（比如，没有 append, insert 等操作）
        # 且不支持通过切片操作动态创建新的 ShareableList  实例。

        shm_a=smm.SharedMemory(size=n)
        # shm_a.buf[:] = bytearray([0]*n)
        # shm_a.buf[:] = [0] * n

        print('shm_a id in main process: {} '.format(id(shm_a)))


        # 主进程的内存空间 和 子进程的内存空间 的考察
        self.global_array=[0]*n
        print('array id in main process: {} '.format(id( self.global_array)) )

        self.global_string='abc'
        print('string id in main process: {} '.format(id(self.global_string)))

        self.global_int=10
        print('int id in main process: {} '.format(id(self.global_int)))


        for i in range(n):

            # p.apply_async(task, args=(cache_name,i)) # apply_async 异步取回结果
            p.apply_async(self.task, args=(cache_list, shm_a ,i))

        print('Waiting for all subprocesses done...')
        p.close()

        p.join()

        print('All subprocesses done.')


        smm.shutdown()

        return cache_list,shm_a

def singleprocess(n):

    cache=[0]*n

    for i in range( n):

        for j in range(i): # 时间复杂度: O(n^2)
            j+=1

        cache[i] =  np.sqrt(np.log(i ** 2 + 1))

    return cache

if __name__=='__main__':

    # n=20
    n = 20000

    # start = time.time()
    #
    # res=singleprocess(n)
    #
    # end = time.time()
    # print(res[:20])
    # print('single processs runs %0.2f seconds.' % ((end - start)))

    # n=20000
    # 13.54s

    start = time.time()

    sol=Solution()
    cache_list,shm_a=sol.multiprocess( n,nthread=2)

    end = time.time()

    print(list(cache_list)[:20]) #  ShareableList 不能切片

    # b=bytes(shm_a.buf[:])
    # print('b:',b)

    b = []
    for i in range(n):
        b.append(shm_a.buf[i])

    print(b[:20])

    # print(sol.global_array[:20])

    print('multi processs runs %0.2f seconds.' % ((end - start)))

    # n=20000
    # nthread=2  time:24.29
    # nthread=4  time: 14.60
    # nthread=8  time:  11.82
    # nthread=16  time: 12.98