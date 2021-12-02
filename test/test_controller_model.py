import sys
import os


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(path)
os.chdir(path)
sys.path.append(path)
from controller_model import NASController


def test_controller_model():
    ctrler = NASController()
    print(ctrler.model)
    arc_sample, log_probs, entropys, skip_penaltys = ctrler()
    print(arc_sample)
    print(log_probs)
    print(entropys)
    print(skip_penaltys)


if __name__ == '__main__':
    test_controller_model()