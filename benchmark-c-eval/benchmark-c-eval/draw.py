import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json
import os
plt.rcParams['font.sans-serif'] = 'Heiti TC'
def plot_radar(data,path,pic_name):
    angles = np.linspace(0, 2 * np.pi, data.shape[0], endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'polar': True})
    col = data.columns.to_list()
    for i in range(data.shape[1]):
        val=data.iloc[:,i].to_list()
        val+=val[:1]
        # if "moss" in col[i]:
        #     color='yellowgreen'
        # elif "chatglm6b" in col[i]:
        #     color = 'pink'
        # elif "chatglm2_6b" in col[i]:
        #     color = 'cadetblue'
        # elif "llama2" in col[i]:
        #     color="salmon"

        ax.plot(angles, val, linewidth=2,label=f"{data.columns[i]}")
        ax.fill(angles, val, alpha=0.25)
    ax.set_title(f"{pic_name}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data.index.to_list())
    plt.legend()
    plt.savefig(path)
    # plt.show()

def split_draw(data,path,map=None):
    #"ntrain=0_cot=False""-------------------------------------------------
    n0nc=[]
    for i in data.columns:
        if "ntrain=0_cot=False" in i :
            n0nc.append(i)
    new_df=data[n0nc]
    if not map:
        pic_path=f"{path}/ntrain=0_cot=False.png"
    else:
        pic_path = f"{path}/{map}_ntrain=0_cot=False.png"
    plot_radar(new_df,pic_path,pic_name="ntrain=0_cot=False")

    #"ntrain=5_cot=False""-------------------------------------------------
    n0nc=[]
    for i in data.columns:
        if "ntrain=5_cot=False" in i :
            n0nc.append(i)
    new_df=data[n0nc]
    if not map:
        pic_path=f"{path}/ntrain=5_cot=False.png"
    else:
        pic_path = f"{path}/{map}_ntrain=5_cot=False.png"
    plot_radar(new_df,pic_path,pic_name="ntrain=5_cot=False")


    #"ntrain=5_cot=True"-------------------------------------------------
    n0nc=[]
    for i in data.columns:
        if "ntrain=5_cot=True" in i :
            n0nc.append(i)
    new_df=data[n0nc]
    if not map:
        pic_path=f"{path}/ntrain=5_cot=True.png"
    else:
        pic_path = f"{path}/{map}_ntrain=5_cot=True.png"
    plot_radar(new_df,pic_path,pic_name="ntrain=5_cot=True")

def mean_draw(data,path,map=None):
    #平均能力"-------------------------------------------------
    df=pd.DataFrame()
    n0nc=[]
    for i in data.columns:
        if "moss" in i :
            n0nc.append(i)
    df1=data[n0nc]
    df['moss_mean']=df1.mean(axis=1)

    n0nc=[]
    for i in data.columns:
        if "chatglm2_6b_text" in i :
            n0nc.append(i)
    df2=data[n0nc]
    df['chatglm2_6b_text_mean']=df2.mean(axis=1)

    n0nc=[]
    for i in data.columns:
        if "chatglm2_6b_chat" in i :
            n0nc.append(i)
    df2=data[n0nc]
    df['chatglm2_6b_chat_mean']=df2.mean(axis=1)

    n0nc=[]
    for i in data.columns:
        if "chatglm6b" in i :
            n0nc.append(i)
    df3=data[n0nc]
    df['chatglm6b_mean']=df3.mean(axis=1)

    n0nc=[]
    for i in data.columns:
        if "llama2_13b_chat" in i :
            n0nc.append(i)
    df2=data[n0nc]
    df['llama2_13b_chat_mean']=df2.mean(axis=1)

    n0nc=[]
    for i in data.columns:
        if "llama2_13b_text" in i :
            n0nc.append(i)
    df2=data[n0nc]
    df['llama2_13b_text_mean']=df2.mean(axis=1)

    n0nc=[]
    for i in data.columns:
        if "lora_llama2" in i :
            n0nc.append(i)
    df2=data[n0nc]
    df['lora_llama2_mean']=df2.mean(axis=1)

    if not map:
        pic_path=f"{path}/mean.png"
    else:
        pic_path = f"{path}/{map}_mean.png"
    plot_radar(df,pic_path,pic_name="avg")

def data_pre(path,split=False,mean=False):
    file_path=os.path.join(path,"res_total","csv")
    radar_path=os.path.join(path,"res_total","radar")
    pic_path=os.path.join(path,"res_total","pic")
    if not os.path.exists(radar_path):
        os.makedirs(radar_path)
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    ls=[]
    file=glob.glob(f"{file_path}/*.csv")
    for  i in file:
        # model_name=i.split('/')[-1].split('_',2)[2][:-4]
        model_name=i.split('/')[-1][:-4]
        df=pd.read_csv(i,index_col=0)
        df=df.rename(columns={"准确率":f"{model_name}"})
        ls.append(df)
    data=pd.concat(ls,axis=1)

    #字典映射
    with open(f'{path}/data/subject_mapping.json') as f:
        js=json.load(f)

    data['subject']=data.index
    data['map']=data['subject'].apply(lambda x: js[x][2])
    data = data.drop('subject', axis=1)
    map_df=data.groupby('map').mean()
    if split:
        split_draw(data, radar_path, map="")
        split_draw(map_df, radar_path, map="map")
    if mean:
        mean_draw(data, radar_path, map="")
        mean_draw(map_df, radar_path, map="map")
    print('执行完毕')





if __name__=="__main__":
    path="/Users/semeron/python/benchmark"
    data_pre(path,split=True,mean=True)


