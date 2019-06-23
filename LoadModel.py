import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from  torch.autograd import Variable
from testDataset import ReadData
from tkinter import *
import tkinter.filedialog
import tkinter.font as tkFont
from PIL import Image,ImageTk


model_path = "./model/last_model.plk"  # 模型的路径
model = torch.load(model_path)  # 加载模型

# 获得dataloader
def getTestData(file):
    labels = ["0"]  # 伪标签
    file_path = []  # 文件路径
    file_path.append(file) # 加入文件
    testset = ReadData(file_list=file_path, labels=labels) # 获得图片的dataset
    # 生成 dataloader 对象,batch_size = 1只取一张图片
    data_test = DataLoader(testset, batch_size=1)
    return data_test  # 返回dataloader对象


# 加载模型进行识别
def test_model(data_set):
    model.eval()  # 转化模型的工作方式
    pred = [] # 识别的结果
    for images, _ in data_set:
        output = model(images) # 识别的结果
        pred.append(output.argmax(dim=1))
    return pred  # 返回识别的结果


# 选择按钮的单击监听器
def callBack():
    # 打开文件选择窗口并选择文件
    filename = tkinter.filedialog.askopenfilename(filetypes=[("JPG",".jpg"),("PNG",".png")])
    # print(filename)
    # 识别的车标集
    label = ['Benz', 'Buick', 'Citroen', 'FAW', 'Fukude', 'Honda', 'Hyundai', 'KIA', 'Lexus', 'Mazda','Unknow']
    dataset = getTestData(filename)  # 获得选择的图片文件的dataloader
    pred = test_model(dataset)  # 加载模型进行识别
    img_open = Image.open(filename)  # 加载图片文件
    img_png = ImageTk.PhotoImage(img_open) # 将图片文件加载入tk中
    pred_num = int(pred[0])  # 获得识别的结果
    # print(label[pred_num])
    text = "识别为:"+label[pred_num] # 将识别的结果和车标名称对应上
    Label(root, text=text,font =Label_Font).pack() # 显示车标名称
    label_img = tkinter.Label(root, image=img_png)
    label_img.pack()  # 显示选择的图片
    root.mainloop()

root = Tk()
Button_Font = tkFont.Font(family="Helvetica",size=36, weight="bold")
Label_Font = tkFont.Font(family="Helvetica",size=24, weight="bold")


def main():
    root.geometry("800x660")
    Button(root, text="选择识别图片", command=callBack, font=Button_Font).pack()
    root.mainloop()


if __name__ == '__main__':
    main()
