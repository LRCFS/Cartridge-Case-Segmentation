import sys
def squared(a,omtrek):
    b = (omtrek/2)-a
    return b
if __name__ == '__main__':
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    sys.stdout.write(str(squared(x,y)))