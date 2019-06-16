import os,shutil


def move(filepath):
    dirpath = "testdata\\"
    shutil.move(filepath,dirpath)


def main():
    files = os.listdir('data')

    for file in files:
        src = 'data\\'
        s = file.strip().split("_")[1]
        s = s.strip().split(".")[0]
        num = int(s)
        if num in range(151, 201):
            src += file
            print(src)
            move(src)


if __name__ == '__main__':
    main()
