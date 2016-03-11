# Alignment-Classifier

Main file- run_alignment_classifier.py .
The json file containing train and test data must be passed to extract training and testing data along with features.

The sentences are then extracted from the json file and complied into a dataframe.

Frame features and POS tags are calculated sentence wise

NOTE: Need to extract just the sentences to a text file and pass it to SEMAFOR as input.Insert the txt.out file obtained from SEMAFOR in calculate_sentence_features.py ,get_frame_feature function


The argument pairs are then formed and the features are calculated.

Same procedure is carried out for test data.Pass the appropriate json file .


Pass the test data,train data and train process name file to classifier
