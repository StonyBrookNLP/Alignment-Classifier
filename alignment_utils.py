import ilp_utils
from word2vec import *
from ilp_utils import *
import json
import pandas as pd
import itertools
import numpy as np
from frame_extraction import *
from operator import itemgetter
import os
import math
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
import subprocess
#function to compute the sentences dataframe from the data.also includes ov1- index where argument and sentence overlap.This helps for finding neighbouring frames

#returns sentences dataframe
def extract_testsen(json_filename):
    data=load_srl_data(json_filename)
    sen_t1=pd.DataFrame(columns=['sen_id','Sentence','Process','arg_id','Arg','role'])
    c=0

    for p in data:

            for i in data[p][1]:

                if data[p][1][i]==[]:
                    #print "Breaking",i
                    continue
                #print data[p]
                for n in data[p][1][i]:
                        #print n
                        sen_t1.loc[c,'Process']=p
                        sen_t1.loc[c,'Arg']=n[1]
                        sen_t1.loc[c,'sen_id']=i
                        sen_t1.loc[c,'arg_id']=int(n[0])
                        id1=(i,int(n[0]))
                        sen_t1.loc[c,'role']=data[p][3][id1][0]
                        for d in data[p][0]:
                            if data[p][0][d]==i:
                                sen_t1.loc[c,'Sentence']=d

                        c=c+1
    #creating replica of arguments so that they can be modified
    sen_t1['Arg_dup']=sen_t1['Arg']

    for i in list(sen_t1.index.get_values()):

            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace('-LRB- ','(')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' -RRB-',')')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' ,',',')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace('-LSB- ','[')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' -RSB-',']')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' \'' , '\'')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' ;',';')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' .','.')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' - ','- ')
            sen_t1.loc[i,'Arg_dup']=sen_t1.loc[i,'Arg_dup'].replace(' :',':')

            sen_t1.loc[i,'o1']=sen_t1.loc[i,'Sentence'].find(sen_t1.loc[i,'Arg_dup'])

    #sen_t1.to_csv('NEWSENDATAQA.csv',sep='\t')
    return sen_t1

def extract_goldsen(json_filename):
    d_gold = json.load(open(json_filename, "r"))
    gold_data =get_gold_data(d_gold)
    sentences=pd.DataFrame(columns=['sen_id','Sentence','Process','Arg','role'])
    n=0
    #getting sentence data
    data=load_srl_data(json_filename)
    s={}
    for p in data:
        s[p]=data[p][4]



    for i in gold_data :

        sentences.loc[n,'sen_id']=i[0]
        sentences.loc[n,'Arg']=i[1]
        sentences.loc[n,'Process']=i[4]

        sentences.loc[n,'role']=gold_data[i]
        ind=sentences.loc[n,'sen_id']
        p=sentences.loc[n,'Process']
        sentences.loc[n,'Sentence']=s[p][ind]

        n=n+1
    #creating replica of arguments so that they can be modified
    sentences['Arg_dup']=sentences['Arg']

    #optional:delete spaces near special characters
    for i in list(sentences.index.get_values()):
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace('-LRB- ','(')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' -RRB-',')')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' ,',',')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace('-LSB- ','[')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' -RSB-',']')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' \'' , '\'')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' ;',';')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' .','.')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' - ','- ')
        sentences.loc[i,'Arg_dup']=sentences.loc[i,'Arg_dup'].replace(' :',':')

        #finding index of argument in sentence
        sentences.loc[i,'o1']=sentences.loc[n,'Sentence'].find(sentences.loc[n,'Arg_dup'])


    return sentences

def get_frame_feature(sen_t1):
    sen_t1['Sentence'].to_csv('/home/sadhana/semafor-semantic-parser/file_2/sentences.txt')
    subprocess.call(['/home/sadhana/semafor-semantic-parser/release/fnParserDriver.sh','/home/sadhana/semafor-semantic-parser/file_2/sentences.txt'])

    frame_test=get_frames('sentences.txt.out')#Insert appropriate semafor output file
    print "Getting sen features "
    for i in list(sen_t1.index.get_values()):
            fs=frame_test[i]


            fs.sort(key=itemgetter(2),reverse=False)
            maxi1=0
            mini1=1000



            le=int(sen_t1.loc[i,'o1'])


            for j in range(0,len(fs)):
                e=int(fs[j][2])
                s=int(fs[j][1])
                if(e > maxi1) and ( e < le):


                        lf=fs[j][0]
                        maxi1=int(fs[j][2])

                else:
                    lf=''

                if(s < mini1) and (s > (le +len(sen_t1.loc[i,'Arg']))):


                        mini1= int(fs[j][1])

                        rf=fs[j][0]
                else:
                    rf=''



            sen_t1.loc[i,'lf1']=lf

            sen_t1.loc[i,'rf1']=rf
    return sen_t1

def get_pos_tag(sen):
    os.environ['CLASSPATH']='STANFORDTOOLSDIR/stanford-postagger-full-2015-12-09/stanford-postagger.jar' #set classpath to pos tagger
    os.environ['STANFORD_MODELS']='STANFORDTOOLSDIR/stanford-postagger-full-2015-12-09/models'
    st = StanfordPOSTagger('/home/sadhana/stanford-postagger-full-2015-12-09/models/english-left3words-distsim.tagger',path_to_jar=
                           '/home/sadhana/stanford-postagger-full-2015-12-09/stanford-postagger.jar')#,path_to_models_jar='/home/sadhana/stanford-postagger-full-2015-12-09/models')

    stanford_dir = st._stanford_jar.rpartition('/')[0]
    stanford_jars = find_jars_within_path(stanford_dir)
    st._stanford_jar = ':'.join(stanford_jars)
    for i in list(sen.index.get_values()):
        t=st.tag(sen.loc[i,'Arg'].split())
        tags=[]
        for j in range(0,len(t)):
            tags.append(t[j][1])
        #print i
        sen.set_value(i,'POStag',tags)
    return sen

def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
def cosine_similarity(x,y):
    cs = np.dot(x,y)/(np.linalg.norm(x)* np.linalg.norm(y))
    score=(cs+1)/2
    return score

def get_rf_cosine_similarity(df):
    w=Word2VecModel()

    for j in list(df.index.get_values()):

        try:
            if type(df.loc[j,'rf1'])==float:
                if math.isnan(df.loc[j,'rf1']) :
                    v1=[]
            else:
                df.loc[j,'rf1']=df.loc[j,'rf1'].replace('_',' ')
                v1=w.get_sent_vector(df.loc[j,'rf1'])
            if type(df.loc[j,'rf2'])==float:
                if math.isnan(df.loc[j,'rf2']):
                    v2=[]
            else:

                    df.loc[j,'rf2']=df.loc[j,'rf2'].replace('_',' ')
                    v2=w.get_sent_vector(df.loc[j,'rf2'])
            if v1!=[] and v2!=[]:
                score=cosine_similarity(v1,v2)
            else:
                score=0
        except ValueError:
            score=0
        df.loc[j,'rf_Cscore']=score
    return df
def get_lf_cosine_similarity(df):
    w=Word2VecModel()

    for j in list(df.index.get_values()):

        try:
            if type(df.loc[j,'lf1'])==float:
                if math.isnan(df.loc[j,'lf1']) :
                    v1=[]
            else:
                df.loc[j,'lf1']=df.loc[j,'lf1'].replace('_',' ')
                v1=w.get_sent_vector(df.loc[j,'lf1'])
            if type(df.loc[j,'lf2'])==float:
                if math.isnan(df.loc[j,'lf2']):
                    v2=[]
            else:

                    df.loc[j,'lf2']=df.loc[j,'lf2'].replace('_',' ')
                    v2=w.get_sent_vector(df.loc[j,'lf2'])
            if v1!=[] and v2!=[]:
                score=cosine_similarity(v1,v2)
            else:
                score=0
        except ValueError:
            score=0
        df.loc[j,'lf_Cscore']=score
    return df

def get_arg_cosine_simialrity(df):
    w=Word2VecModel()
    for j in list(df.index.get_values()):
        try:
                v1=w.get_sent_vector(df.loc[j,'Arg1'])
                v2=w.get_sent_vector(df.loc[j,'Arg2'])
                score=cosine_similarity(v1,v2)
        except ValueError:
                score=np.nan



        df.loc[j,'Cscore']=score
    return df
def get_entailment_score(df):
    for j in list(df.index.get_values()):

        df.loc[j,'Escore']=get_similarity_score(df.loc[j,'Arg1'],df.loc[j,'Arg2'])
    return df

def get_pos_similarity(df):
    for i in list(df.index.get_values()):
        df.loc[i,'POSsim']=jaccard_similarity(df.loc[i,'POStag1'],df.loc[i,'POStag2'])

def get_arg_pairs(sentences):
    df=pd.DataFrame(columns=['Process','Arg1','Arg2','Sentence1','Sentence2','Role1','Role2','True_label'])
    m=0
    processes=list(set(sentences['Process']))
    for p in processes:
        for ac in itertools.combinations(sentences[sentences['Process']==p].index.tolist(),2):


                        df.loc[m,'Process']=p
                        df.loc[m,'Arg1']=sentences.loc[ac[0],'Arg']
                        df.loc[m,'Arg2']=sentences.loc[ac[1],'Arg']
                        df.loc[m,'Sentence1']=sentences.loc[ac[0],'Sentence']
                        df.loc[m,'Sentence2']=sentences.loc[ac[1],'Sentence']
                        df.loc[m,'Role1']=sentences.loc[ac[0],'Role']
                        df.loc[m,'Role2']=sentences.loc[ac[1],'Role']
                        if df.loc[m,'Role1']==df.loc[m,'Role2']:
                            df.loc[m,'True_label']=1
                        else:
                            df.loc[m,'True_label']=0

                        m=m+1

    return df
def merge_sen_df(sentences,df):
    for i in list(df.index.get_values()):
        print i
        arg=df.loc[i,'Arg1']
        y=sentences[(sentences['Arg']==arg) & (sentences['Process']==df.loc[i,'Process']) ]
        z=sentences[(sentences['Arg']==df.loc[i,'Arg2']) & (sentences['Process']==df.loc[i,'Process']) ]
        df.loc[i,'Sentence1']=y['Sentence'].values[0]
        df.loc[i,'Sentence2']=z['Sentence'].values[0]
        df.loc[i,'lf1']=y['lf1'].values[0]
        df.loc[i,'rf1']=y['rf1'].values[0]
        df.loc[i,'lf2']=z['lf1'].values[0]
        df.loc[i,'rf2']=z['rf1'].values[0]
        df.loc[i,'o1']=y['o1'].values[0]
        df.loc[i,'o2']=z['o1'].values[0]
        df.set_value(i,'POStag1',y['POStag'].values[0])

        df.set_value(i,'POStag2',z['POStag'].values[0])
    return df

def plot_precision_yield(plot_data):
    srl_plot_df = plot_data
    srl_plot_df = srl_plot_df.iloc[10:]

    # plot size
    plt.rc('figure', figsize=(18,12))

    # plot lines
    plt.plot(srl_plot_df.index, srl_plot_df.precision, label=r'POS', linewidth=3)


    # configure plot
    plt.tick_params(axis='both', which='major', labelsize=50)
    plt.xlabel('Recall', fontsize=50)
    plt.ylabel('Precison', fontsize=50)
    plt.xlim([0, 1])
    plt.ylim([0, 1.005])
    plt.legend(loc='lower right', handlelength=3, prop={'size':45}) #borderpad=1.5, labelspacing=1.5,
    plt.tight_layout()
    plt.show()

def plot_pr_overall_concatenated(srl_all_data):
    """Plots overall precision recall after joining data from all the folds"""
    # concatenate data from all folds into a single dictionary which has
    # as key (sentence_id, start_index, end_index)
    # as value (gold_role, (predicted_role, prediction_score))
    # (i.e all fold id based separation is taken off).
    sorted_srl_correct = sorted(srl_all_data.items(), key=lambda x: x[1][1][1], reverse=True)

    srl_yield = []
    gold_role_total = 0
    gold_role_predicted = 0
    total_role_predicted = 0

    for x in sorted_srl_correct:
        key, data = x
        gold_role, srl_data = data
        srl_role, srl_score = srl_data

        gold_role_total += 1
        if srl_role == gold_role:
            gold_role_predicted += 1

        total_role_predicted += 1

        if gold_role_predicted != 0 and total_role_predicted != 0:
            precision = gold_role_predicted/float(total_role_predicted)
        else:
            precision = 0
        srl_yield.append((gold_role_predicted, gold_role_total, precision))

    srl_df = pd.DataFrame(srl_yield)
    srl_df.columns = ['yield', 'total_predicted', 'precision']
    srl_df['recall'] = srl_df['yield']/max(srl_df.total_predicted.tolist())
    srl_yield_df = srl_df.set_index(['recall'])
    srl_yield_df = srl_yield_df['precision']
    srl_plot_df = pd.DataFrame(srl_yield_df)

    # call plot function
    return srl_plot_df