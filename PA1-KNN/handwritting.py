import struct
import threading
import collections as cl
import numpy as np
import tkinter as tk
import tkinter.filedialog
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

run_full = False

global atest,atest_label
global atrain,atrain_label
global count,right
count = 0
right = 0
global fig,ax
#只完成了K=3的情况
#runfull为true的时候，不显示每次测试的图片，只输出测试后的准确率；runfull为false的时候，进入演示模式，点按钮进行单次测试
def begin_test():
    global count,fig,ax,right
    if(run_full):
        for c in range(len(atest_label)):
            distance = []
            for k in atrain:
                #将测试对象atest[count]与样例集中的每个样例计算距离，此处为欧几里得距离（norm函数式求范数，默认为二范数）
                distance.append(np.linalg.norm(atest[count]-k))
            result_index = []
            result = []
            distance_backup = distance.copy()
            for j in  range(0,3):
                #寻找距离list中的最小值，找到对应的index并记录，然后将距离修改为极大值，以此找到第二小的值
                #因K=3，找到三个即可
                index = distance.index(min(distance))
                result_index.append(index)
                result.append(atrain_label[index])
                distance[index] = float('inf')
            prediction = 0
            #找到三个值中出现次数最多的值，如果最多出现次数为1，则退化为K=1，找到距离中最小的一个
            if(cl.Counter(result).most_common(1)[0][1]==1):
                prediction = atrain_label[distance_backup.index(min(distance_backup))]
            else:
                prediction = cl.Counter(result).most_common(1)[0][0]
            #判断预测是否正确
            if(prediction == atest_label[count]):
                right=right+1
            count = count + 1
            p = '%.3f' % (float(right)/count)
            print(p)
        return
    else:  #演示模式基本与上面相同，只是使用matplotlib进行绘图，并且每次只处理一个
        begin_button['text']='下一个'
        distance = []
        for k in atrain:
            distance.append(np.linalg.norm(atest[count]-k))
        result_index = []
        result = []
        distance_backup = distance.copy()
        for j in  range(0,3):
            index = distance.index(min(distance))
            result_index.append(index)
            result.append(atrain_label[index])
            distance[index] = float('inf')
        prediction = 0
        if(cl.Counter(result).most_common(1)[0][1]==1):
            prediction = atrain_label[distance_backup.index(min(distance_backup))]
        else:
            prediction = cl.Counter(result).most_common(1)[0][0]
        if(prediction == atest_label[count]):
            right=right+1
        count = count + 1
        p = '%.3f' % (float(right)/count)
        label_result['text']=str(right)+'/'+str(count)+' | '+p
        if(count==1):
            fig, ax = plt.subplots(
            nrows=1,
            ncols=4,
            sharex=True,
            sharey=True)
            ax = ax.flatten()
        fig.suptitle('Result: '+str(prediction),fontsize=16)
        for i in range(0,4):
            if(i==0):
                img = atest[count-1].reshape(28, 28)
                ax[0].imshow(img, cmap='Greys', interpolation='nearest')
                ax[0].set_title('test: '+str(atest_label[count-1]))
            else:
                img = atrain[result_index[i-1]].reshape(28, 28)
                ax[i].imshow(img, cmap='Greys', interpolation='nearest')
                ax[i].set_title(str(atrain_label[result_index[i-1]]))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
    
    
#数据的导入
def import_train():
    global atrain
    train_filename = tk.filedialog.askopenfilename(initialdir='./')
    train = open(train_filename,'rb')
    magic_num,sample_total,row,column = struct.unpack('>IIII',train.read(16))
    atrain = np.fromfile(train,dtype=np.uint8).reshape(sample_total,784)
    train.close()

def import_train_label():
    global atrain_label
    train_label_filename = tk.filedialog.askopenfilename(initialdir='./')
    train_label = open(train_label_filename,'rb')
    magic_num,num = struct.unpack('>II',train_label.read(8))
    atrain_label = np.fromfile(train_label,dtype=np.uint8)
    train_label.close()

def import_test():
    global atest
    test_filename = tk.filedialog.askopenfilename(initialdir='./')
    test = open(test_filename,'rb')
    magic_num,num_total,row,column = struct.unpack('>IIII',test.read(16))
    atest = np.fromfile(test,dtype=np.uint8).reshape(num_total,784)
    test.close()

def import_test_label():
    global atest_label
    test_label_filename = tk.filedialog.askopenfilename(initialdir='./')
    test_label = open(test_label_filename,'rb')
    magic_num,num = struct.unpack('>II',test_label.read(8))
    atest_label = np.fromfile(test_label,dtype=np.uint8)
    test_label.close()


#图形界面部分
main_window = tk.Tk()        #实例化出一个父窗口
main_window.title('KNN Image Recognition')
import_train_data = tk.Button(text='导入样例数据',command=import_train)
import_train_label = tk.Button(text='导入样例标签',command=import_train_label)
btest_data = tk.Button(text='导入测试数据',command=import_test)
btest_label = tk.Button(text='导入测试标签',command=import_test_label)
begin_button = tk.Button(text='开始',command=begin_test)
import_train_data.grid(column=0,row=0,sticky=tk.S,padx=5,pady=5)
import_train_label.grid(column=1,row=0,sticky=tk.S,padx=5,pady=5)
btest_data.grid(column=0,row=1,sticky=tk.N)
btest_label.grid(column=1,row=1,sticky=tk.N)
begin_button.grid(column=0,row=2)
label_copyright = tk.Label(main_window,text='@Xinrea\ngithub.com/Xinrea')
label_copyright.grid(column=0,row=6,columnspan=5)
label_result = tk.Label(main_window,text='0/0')
label_result.grid(column=1,row=2)
main_window.mainloop()    #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示

#window closed
