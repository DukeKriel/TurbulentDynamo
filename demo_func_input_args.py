import os
import numpy as np

os.system("clear")  # clear terminal window

class myClass():
    def __init__(self, a1, b1, a2, b2):
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

def randData1():
    a1 = np.random.randint(5, size=10)
    b1 = np.random.randint(5, size=10)
    args = {"a1":a1, "b1":b1}
    return args

def randData2():
    a2 = np.random.randint(5, size=10)
    b2 = np.random.randint(5, size=10)
    args = {"a2":a2, "b2":b2}
    return args

args1 = randData1()
args2 = randData2()

tmp_obj = myClass(**args1, **args2)
