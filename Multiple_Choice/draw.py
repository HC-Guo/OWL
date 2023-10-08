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
        ax.plot(angles, val, linewidth=2,label=f"{'_'.join(data.columns[i].split('_')[:-2])}")
        ax.fill(angles, val, alpha=0.25)
    ax.set_title(f"{pic_name}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(angles[:-1])
    print(data.index.to_list())
    ax.set_xticklabels(data.index.to_list())
    plt.legend()
    plt.savefig(path)
    # plt.show()

def split_draw(data,path):
    #"ntrain=0_cot=False""-------------------------------------------------
    n0nc=[]
    for i in data.columns:
        if "ntrain=0_cot=False" in i :
            n0nc.append(i)
    if n0nc:
        new_df=data[n0nc]
        pic_path=f"{path}/ntrain=0_cot=False.png"
        plot_radar(new_df,pic_path,pic_name="ntrain=0_cot=False")

    #"ntrain=5_cot=False""-------------------------------------------------
    n0nc=[]
    for i in data.columns:
        if "ntrain=5_cot=False" in i :
            n0nc.append(i)
    if n0nc:
        new_df=data[n0nc]
        pic_path=f"{path}/ntrain=5_cot=False.png"
        plot_radar(new_df,pic_path,pic_name="ntrain=5_cot=False")

    #"ntrain=5_cot=True"-------------------------------------------------
    n0nc=[]
    for i in data.columns:
        if "ntrain=5_cot=True" in i :
            n0nc.append(i)
    if n0nc:
        new_df=data[n0nc]
        pic_path=f"{path}/ntrain=5_cot=True.png"
        plot_radar(new_df,pic_path,pic_name="ntrain=5_cot=True")


def main(path):
    model=os.listdir(os.path.join(path,'result'))
    radar_path = os.path.join(path, "res_total", "radar")
    if not os.path.exists(radar_path):
        os.makedirs(radar_path)
    ls=[]
    for i in model:
        if i in ['.DS_Store','','llama2_13b']:
            continue
        ntrain=os.listdir(os.path.join(path,"result",f"{i}"))
        for j in ntrain:
            if j=='.DS_Store':
                continue
            cot=os.listdir(os.path.join(path,"result",f"{i}",f"{j}"))
            for k in cot:
                if k == '.DS_Store':
                    continue
                file=glob.glob(os.path.join(path,"result",f"{i}",f"{j}",f"{k}",'*.csv'))
                if len(file)!=1:
                    print('文件路径个数不为1')
                    exit()
                pic_name=f"{i}_{j}_{k}"
                df=pd.read_csv(file[0])[['correct','category']]
                # mean=df['correct'].mean()
                # df_=pd.DataFrame(data={'correct':[mean]},index=['mean'])
                df = df.groupby('category').mean().T
                df["ALL"]=df.mean(axis=1)["correct"]
                df=df.T
                df=df.rename(columns={"correct": f"{i}_{j}_{k}"})
                df=np.round(df*100,2)
                ls.append(df)
    df=pd.concat(ls,axis=1)
    # print(df)
    split_draw(df, radar_path)




if __name__=="__main__":
    path=os.path.dirname(__file__)
    main(path)

