#!/bin/bash

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.98 --budget 250 --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.95 --budget 250 --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.9 --budget 250 --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.8 --budget 250 --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.7 --budget 250 --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.6 --budget 250 --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0 --budget 250 --substitute_threshold 0.7
echo "-------------------------   budget 1k    --------------------------------"

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.98 --budget 1k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.95 --budget 1k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.9 --budget 1k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.8 --budget 1k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.7 --budget 1k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.6 --budget 1k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0 --budget 1k --substitute_threshold 0.7

echo "-------------------------   budget 2k    --------------------------------"

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.98 --budget 2k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.95 --budget 2k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.9 --budget 2k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.8 --budget 2k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.7 --budget 2k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.6 --budget 2k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0 --budget 2k --substitute_threshold 0.7

echo "-------------------------   budget 4k    --------------------------------"

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.98 --budget 4k --substitute_threshold 0.7 

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.95 --budget 4k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.9 --budget 4k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.8 --budget 4k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.7 --budget 4k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0.6 --budget 4k --substitute_threshold 0.7

python Step4_Proposed_Method_v2.py --cossim substitute --threshold 0 --budget 4k --substitute_threshold 0.7



#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 1 --susbtitute_and_cossim 0 --scheduler 1 --threshold 0  #cossim

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --method AL --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 0  #substitute

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 9 #both

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 95

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 96

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 97

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 98

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 8

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 85

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 75

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 7

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --method AL --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 65

#python Step5_TrainAndEval_SusbtituteModel.py --epoch 10 --method AL --lr 0.01 --momentum 0.9 --whole_data_labeled_with_cossim 0 --susbtitute_and_cossim 1 --scheduler 1 --threshold 6














