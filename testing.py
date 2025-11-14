def main():
    d = [1]
    foo(d)
    print(d)

def foo(d):
    f = [100]
    d[0] = f
    del f

if __name__ == '__main__':
    main()
