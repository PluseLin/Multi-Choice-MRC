'''
模块主要工作：
处理RACE数据集
'''

import glob
import os
import json

def preprocess(task,level=['high','middle']):
    print('Preprocessing the dataset ' + task + '...')
    dataset_names = ['train', 'dev', 'test']
    level_names = level
    q_id = 0
    label_dict = {'A':0, 'B':1,'C':2, 'D':3}
    for dataset_name in dataset_names:
        data_all = []
        for level_name in level_names:
            #拼接地址./data/task/RACE/train/high，并从中找到所有的txt文件
            path = os.path.join('data','RACE', dataset_name, level_name)
            filenames = glob.glob(path+'/*txt')
            for filename in filenames:
                #逐一读取处理，txt文件是json格式
                with open(filename, 'r', encoding='utf-8') as fpr:
                    data_raw = json.load(fpr)
                    #分词
                    #artical = [ word_tokenize(s.strip()) for s in sent_tokenize(data_raw['article']) ]
                    artical = data_raw['article']
                    #每个答案、文章和问题配对
                    for i in range(len(data_raw['answers'])):
                        instance = {}
                        instance['race_id'] = q_id
                        instance['article'] = artical
                        instance['question'] = data_raw['questions'][i]
                        instance['label'] = label_dict[ data_raw['answers'][i] ]
                        instance['filename'] = filename
                        # instance['options'] = [ word_tokenize( option ) for option in data_raw['options'][i] ]
                        _options=data_raw['options'][i] 
                        instance['option_0']=_options[0]
                        instance['option_1']=_options[1]
                        instance['option_2']=_options[2]
                        instance['option_3']=_options[3]
                        q_id += 1
                        data_all.append(instance)
                        if len(data_all) % 2000==0:
                            print(len(data_all))
        #写入到json中
        with open(os.path.join('data','sequence', dataset_name)+'.json', 'w', encoding='utf-8') as fpw:
            json.dump(data_all, fpw)

if __name__ == '__main__':
    preprocess('race')
