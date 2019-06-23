# encoding=utf-8
import os, shutil

'''
该文件是将data里面的数据分割成训练集和测试集
训练集：取每一类前150张图片
测试集：取每一类的后50张图片
'''


# 移动函数
def move(filepath):
    dirpath = "testdata\\"  # 移动的目标文件夹路径
    shutil.move(filepath, dirpath)  # 移动文件


def main():
    files = os.listdir('data')  # 获得data文件夹中的文件列表
    for file in files: # 遍历文件列表
        src = 'data\\'
        # 1_124.jpg => 124.jpg => 124
        s = file.strip().split("_")[1] # 获得文件的标号
        s = s.strip().split(".")[0] #
        num = int(s)  # 转换成数字
        if num in range(151, 201):  # 在150-200区间内移动到test文件夹
            src += file # 组装源文件路径
            # print(src)
            move(src) # 调用移动函数


if __name__ == '__main__':
    main()
