import pandas as pd


from alignment_utils import *
from classifier import *
#Construct training data for 5 folds
for fold_no in xrange(5):
       print " Getting features for training data in Fold "+str((fold_no+1))
       fold_path = "/home/slouvan/NetBeansProjects/ILP/data/cross-val/fold-"+str((fold_no+1))
       #fold_path = "/home/slouvan/NetBeansProjects/ILP/data/cross-val-small/fold-3"
       srl_out_file_path = join(fold_path, 'test', 'test.srlout.json')
       gold_sentence_df=extract_goldsen(srl_out_file_path)
       gold_sentence_df=get_frame_feature(gold_sentence_df)
       gold_sentence_df=get_pos_tag(gold_sentence_df)
       gold_df=get_arg_pairs(gold_sentence_df)
       gold_df=merge_sen_df(gold_sentence_df,gold_df)
       gold_df=get_arg_cosine_simialrity(gold_df)
       gold_df=get_lf_cosine_similarity(gold_df)
       gold_df=get_rf_cosine_similarity(gold_df)
       gold_df=get_entailment_score(gold_df)
       gold_df=get_pos_similarity(gold_df)


       if int(fold_no+1)==1 :
            train_df=gold_df
       else:
           train_df=pd.concat([train_df,gold_df]).reset_index()





#Construct test dataframe
for fold_no in xrange(5):
    print " Getting features for testing data in Fold "+str((fold_no+1))
    fold_path = "/home/slouvan/NetBeansProjects/ILP/data/cross-val/fold-"+str((fold_no+1))
    #fold_path = "/home/slouvan/NetBeansProjects/ILP/data/cross-val-small/fold-3"
    srl_predict_file_path = join(fold_path, 'test', 'test.srlpredict.json')
    train_process_file_path=join(fold_path, 'train', 'train_process_name')
    test_sentence_df=extract_testsen(srl_predict_file_path)
    test_sentence_df=get_frame_feature(test_sentence_df)
    test_sentence_df=get_pos_tag(test_sentence_df)

    test_df=get_arg_pairs(test_sentence_df)

    test_df=merge_sen_df(test_sentence_df,test_df)
    test_df=get_arg_cosine_simialrity(test_df)
    test_df=get_lf_cosine_similarity(test_df)
    test_df=get_rf_cosine_similarity(test_df)

    test_df=get_entailment_score(test_df)
    test_df=get_pos_similarity(test_df)

    #classification,precision,f1 score is printed and final dataframe with results is  returned
    print "Training and testing  "+str((fold_no+1))
    result_df=classifier(test_df,train_df,train_process_file_path)

    if int(fold_no+1)==1 :
            final_results_df=result_df
    else:
           final_results_df=pd.concat([final_results_df,result_df]).reset_index()
#preparing plot data
plot_data = final_results_df[['True_label', 'Classification_result', 'Probability of result']]
plot_data.columns = ['gold_role', 'srl_role', 'srl_score']

d = {}
for rid, rdata in plot_data.iterrows():
    grole = rdata['gold_role']
    srole = rdata['srl_role']
    sscore = rdata['srl_score']
    d[rid] = (grole, (srole, sscore))

p_df=plot_pr_overall_concatenated(d)
plot_precision_yield(p_df)