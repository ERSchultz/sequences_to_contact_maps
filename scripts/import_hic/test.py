import numpy as np


def main():
    bad = np.array([0,0,0,1,0,1,1,0]).astype(bool)
    ind = np.arange(0, len(bad))
    print(ind)
    print(bad)
    arg_rev = np.argmax(np.flip(bad))
    print(arg_rev)
    print(ind[-(arg_rev+1)])

if __name__ == '__main__':
    main()
