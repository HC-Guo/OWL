import numpy as np
import pandas as pd

import os
import glob

class Data():
    def __init__(self,path,ntrain,save_result_dir=None):
        self.path=path
        self.ntrain=ntrain
        self.save_result_dir=save_result_dir
        self.choices=["A", "B", "C", "D"]
        self.val_dict = self.read_val()
        self.dev_dict = self.read_dev()

    def read_val(self,path="data/val/*.csv"):
        val_dict={}
        search=os.path.join(self.path,path)
        # search=self.path+"/data/val/*.csv"
        file=glob.glob(search)
        if not file :
            print("验证集路径错误")
            return
        subject=[i.split('/')[-1].split('.')[0] for i in file]
        cnt=0
        for i in subject:
            path=os.path.join(self.path, 'data','val', f'{i}.csv')
            df=pd.read_csv(path)
            df["choices"]=df.apply(lambda x :f"A.{x['A']}\nB.{x['B']}\nC.{x['C']}\nD.{x['D']}\n",axis=1)
            val_dict[i]=df[['question','choices','answer']]
            cnt+=df.shape[0]
            print(f"学科{i}处理完毕，有{df.shape[0]}条数据")
        print(f'读取完毕,共计{cnt}条数据')
        return  val_dict

    def read_dev(self,path="data/shot/*.csv"):
        dev_dict={}
        search = os.path.join(self.path, path)
        # search=self.path+"/data/dev/*.csv"
        file=glob.glob(search)
        if not file :
            print("开发集集路径错误")
            return
        subject=[i.split('/')[-1].split('.')[0] for i in file]
        for i in subject:
            path=os.path.join(self.path, 'data/shot', f'{i}.csv')
            df=pd.read_csv(path)
            df["choices"]=df.apply(lambda x :f"A.{x['A']}\nB.{x['B']}\nC.{x['C']}\nD.{x['D']}\n",axis=1)
            dev_dict[i]=df[['question','choices','answer']]
        print('开发集读取完毕')
        return  dev_dict



    def preprocess(self,role=False):
        prompt_dict={}
        if self.ntrain==0:
            if not role:
                for subject in self.val_dict:
                    prompt_dict[subject] = {}
                    prompt = f"你是一个中文人工智能助手，以下是运维领域考试的单项选择题，请选出其中的正确答案。\n"
                    self.val_dict[subject]['question'] = self.val_dict[subject]['question'].apply(lambda x:prompt+"问题：\n"+x)
                    self.val_dict[subject]['input']=self.val_dict[subject]['question']+'\n'+self.val_dict[subject]['choices']+"答案："
                    self.val_dict[subject]['output']=self.val_dict[subject]['answer']
                    prompt_dict[subject]['shot']=pd.DataFrame(columns=['input', 'output'])
                    prompt_dict[subject]['origin'] = self.val_dict[subject][['input','output']]
                return prompt_dict
            else:
                for subject in self.val_dict:
                    prompt_dict[subject] = {}
                    prompt = f"你是一个中文人工智能助手，以下是运维领域考试的单项选择题，请选出其中的正确答案。\n"
                    self.val_dict[subject]['question'] = self.val_dict[subject]['question'].apply(lambda x:prompt+x)
                    self.val_dict[subject]['input']=self.val_dict[subject]['question']+'\n'+self.val_dict[subject]['choices']
                    self.val_dict[subject]['output']=self.val_dict[subject]['answer']
                    prompt_dict[subject]['shot']=pd.DataFrame(columns=['input', 'output'])
                    prompt_dict[subject]['origin'] = self.val_dict[subject][['input','output']]
                return prompt_dict
        else:
            if not role:
                for subject in self.val_dict:
                    prompt_dict[subject] = {}
                    prompt = f"你是一个中文人工智能助手，以下是中国运维领域考试的单项选择题，请选出其中的正确答案。"
                    if self.ntrain > len(self.dev_dict[subject]):
                        self.ntrain = len(self.dev_dict[subject])
                    self.dev_dict[subject]['input'] = self.dev_dict[subject].apply(
                        lambda x: "\n问题：\n" + x['question'] + '\n' + x['choices'] + '答案：' + x['answer'], axis=1)
                    for i in range(self.ntrain):
                        prompt = prompt + self.dev_dict[subject]['input'].iloc[i]
                    self.val_dict[subject]['input'] = prompt + "\n问题\n" + self.val_dict[subject]['question'] + '\n' + \
                                                      self.val_dict[subject]['choices'] + "答案："
                    self.val_dict[subject]['output'] = self.val_dict[subject]['answer']
                    prompt_dict[subject]['shot'] = pd.DataFrame(columns=['input', 'output'])
                    prompt_dict[subject]['origin'] = self.val_dict[subject][['input', 'output']]
                return prompt_dict
            else:
                for subject in self.val_dict:
                    prompt_dict[subject]={}
                    prompt = f"你是一个中文人工智能助手，以下是中国运维领域考试的单项选择题，请选出其中的正确答案。\n"
                    if self.ntrain > len(self.dev_dict[subject]):
                        self.ntrain = len(self.dev_dict[subject])
                    self.dev_dict[subject]['input'] = self.dev_dict[subject].apply(lambda x: x['question'] + '\n' + x['choices'] , axis=1)
                    self.dev_dict[subject]['output'] = self.dev_dict[subject]['answer']
                    df=self.dev_dict[subject][['input','output']].iloc[:self.ntrain,:]
                    prompt_dict[subject]['shot']=df
                    self.val_dict[subject]['input'] =  self.val_dict[subject]['question'] + '\n' + self.val_dict[subject]['choices']
                    self.val_dict[subject]['output'] = self.val_dict[subject]['answer']
                    prompt_dict[subject]['origin'] = self.val_dict[subject][['input','output']]
                return prompt_dict



if __name__=="__main__":
    data=Data(path='/Users/semeron/python/benchmark',
              ntrain=5)
    a=data.preprocess(role=True)
    b=1



