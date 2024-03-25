import time
import collections

from dataclasses import dataclass
from typing import Any,Optional,List

@dataclass
class DataClassTuple:
    x:     Optional[float]
    y:     Optional[float]
   

# Define a simple namedtuple globally
GlobalNamedTuple = collections.namedtuple('GlobalNamedTuple', ['x', 'y'])

def test_global_namedtuple():
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < .0167:
        point = GlobalNamedTuple(x=1, y=2)
        #x,y=1,1
        iterations += 1
    return iterations

def test_local_namedtuple():
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < .0167:
        LocalNamedTuple = collections.namedtuple('LocalNamedTuple', ['x', 'y'])
        point = LocalNamedTuple(x=1, y=2)
        iterations += 1
    return iterations

def test_dataclass_tuple():
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < .0167:
        point = DataClassTuple(x=1, y=2)
        #x,y=1,1
        iterations += 1
    return iterations

def test_tuple():
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < .0167:
        point = (1, 2)
        iterations += 1
    return iterations

def test_dict():
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < .0167:
        point = {'x':1, 'y':2}
        iterations += 1
    return iterations

if __name__ == "__main__":

    # Measure the time taken and the number of iterations for defining a global dataclass
    class_iterations = test_dataclass_tuple()
    print("\nDataClass Performance Test:")
    print(f"Time taken for dataclass: 1 frame at 60fps")
    print(f"Number of iterations for dataclass: {class_iterations}")

    # Measure the time taken and the number of iterations for defining a global namedtuple
    global_iterations = test_global_namedtuple()
    print(f"\nGlobal Namedtuple Performance Test:")
    print(f"Time taken for global namedtuple: 1 frame at 60fps")
    print(f"Number of iterations for global namedtuple: {global_iterations}")

    # Measure the time taken and the number of iterations for defining a local namedtuple
    local_iterations = test_local_namedtuple()
    print("\nLocal Namedtuple Performance Test:")
    print(f"Time taken for local namedtuple: 1 frame at 60fps")
    print(f"Number of iterations for local namedtuple: {local_iterations}")

    # Measure the time taken and the number of iterations for defining a local tuple
    local_iterations = test_tuple()
    print("\nLocal tuple Performance Test:")
    print(f"Time taken for local tuple: 1 frame at 60fps")
    print(f"Number of iterations for local tuple: {local_iterations}")
    
    # Measure the time taken and the number of iterations for defining a local tuple
    local_iterations = test_dict()
    print("\nLocal dict Performance Test:")
    print(f"Time taken for  dict: 1 frame at 60fps")
    print(f"Number of iterations for dict: {local_iterations}")
