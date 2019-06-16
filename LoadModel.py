import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from  torch.autograd import Variable
from testDataset import ReadData
from tkinter import *
import tkinter.filedialog
import tkinter.font as tkFont
from PIL import Image,ImageTk


model_path = "./model/9_model.plk"
model = torch.load(model_path)


def getTestData(file):

    labels = ["0"]
    file_path = []
    file_path.append(file)

    testset = ReadData(file_list=file_path,labels=labels)
    data_test = DataLoader(testset, batch_size=1)
    return data_test

def test_model(data_set):
    model.eval()
    pred = []
    for images, labels in data_set:
        output = model(images)
        pred.append(output.argmax(dim=1))
    return pred


def callBack():
    filename = tkinter.filedialog.askopenfilename(filetypes=[("JPG",".jpg"),("PNG",".png")])
    print(filename)
    label = ['Benz', 'Buick', 'Citroen', 'FAW', 'Fukude', 'Honda', 'Hyundai', 'KIA', 'Lexus', 'Mazda','Unknow']
    dataset = getTestData(filename)
    pred = test_model(dataset)
    img_open = Image.open(filename)
    img_png = ImageTk.PhotoImage(img_open)
    pred_num = int(pred[0])
    print(label[pred_num])
    text = "识别为:"+label[pred_num]
    Label(root, text=text,font =Label_Font).pack()
    label_img = tkinter.Label(root, image=img_png)
    label_img.pack()
    root.mainloop()

root = Tk()
Button_Font = tkFont.Font(family="Helvetica",size=36, weight="bold")
Label_Font = tkFont.Font(family="Helvetica",size=24, weight="bold")

def main():

    root.geometry("800x660")
    Button(root, text="选择识别图片", command=callBack,font = Button_Font).pack()
    root.mainloop()


if __name__ == '__main__':
    main()