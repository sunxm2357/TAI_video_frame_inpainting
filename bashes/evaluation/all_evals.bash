#!/bin/bash

KTH=$1
HMDB=$2
UCF=$3

# eval on kth
bash bashes/evaluation/kth_TAI/test_KTH.bash 5 10 $KTH
bash bashes/evaluation/kth_TW/test_KTH.bash 5 10 $KTH
bash bashes/evaluation/kth_SA/test_KTH.bash 5 10 $KTH
bash bashes/evaluation/kth_mcnet/test_KTH.bash 5 10 $KTH
bash bashes/evaluation/kth_trivial_baselines/repeat_P.bash 5 10 $KTH
bash bashes/evaluation/kth_trivial_baselines/repeat_F.bash 5 10 $KTH
bash bashes/evaluation/kth_trivial_baselines/SA_P_F.bash 5 10 $KTH
bash bashes/evaluation/kth_trivial_baselines/TW_P_F.bash 5 10 $KTH

#eval on hmdb
bash bashes/evaluation/hmdb_TAI/test_HMDB.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_TW/test_HMDB.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_SA/test_HMDB.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_mcnet/test_HMDB.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_trivial_baselines/repeat_P.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_trivial_baselines/repeat_F.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_trivial_baselines/SA_P_F.bash 4 5 $HMDB
bash bashes/evaluation/hmdb_trivial_baselines/TW_P_F.bash 4 5 $HMDB

#eval on ucf
bash bashes/evaluation/ucf_TAI/test_UCF.bash 4 5 $UCF
bash bashes/evaluation/ucf_TW/test_UCF.bash 4 5 $UCF
bash bashes/evaluation/ucf_SA/test_UCF.bash 4 5 $UCF
bash bashes/evaluation/ucf_mcnet/test_UCF.bash 4 5 $UCF
bash bashes/evaluation/ucf_trivial_baselines/repeat_P.bash 4 5 $UCF
bash bashes/evaluation/ucf_trivial_baselines/repeat_F.bash 4 5 $UCF
bash bashes/evaluation/ucf_trivial_baselines/SA_P_F.bash 4 5 $UCF
bash bashes/evaluation/ucf_trivial_baselines/TW_P_F.bash 4 5 $UCF

#eval for ablation study
bash bashes/evaluation/kth_TWI/test_KTH.bash 5 10 $KTH

#eval for context
bash bashes/evaluation/kth_TAI/test_KTH.bash 2 10 $KTH
bash bashes/evaluation/kth_TAI/test_KTH.bash 3 10 $KTH
bash bashes/evaluation/kth_TAI/test_KTH.bash 4 10 $KTH

# eval for variant outputs
bash bashes/evaluation/kth_TAI/test_KTH.bash 5 6 $KTH
bash bashes/evaluation/kth_TAI/test_KTH.bash 5 7 $KTH
bash bashes/evaluation/kth_TAI/test_KTH.bash 5 8 $KTH
bash bashes/evaluation/kth_TAI/test_KTH.bash 5 9 $KTH
bash bashes/evaluation/kth_TW/test_KTH.bash 5 6 $KTH
bash bashes/evaluation/kth_TW/test_KTH.bash 5 7 $KTH
bash bashes/evaluation/kth_TW/test_KTH.bash 5 8 $KTH
bash bashes/evaluation/kth_TW/test_KTH.bash 5 9 $KTH
bash bashes/evaluation/kth_SA/test_KTH.bash 5 6 $KTH
bash bashes/evaluation/kth_SA/test_KTH.bash 5 7 $KTH
bash bashes/evaluation/kth_SA/test_KTH.bash 5 8 $KTH
bash bashes/evaluation/kth_SA/test_KTH.bash 5 9 $KTH
bash bashes/evaluation/kth_mcnet/test_KTH.bash 5 6 $KTH
bash bashes/evaluation/kth_mcnet/test_KTH.bash 5 7 $KTH
bash bashes/evaluation/kth_mcnet/test_KTH.bash 5 8 $KTH
bash bashes/evaluation/kth_mcnet/test_KTH.bash 5 9 $KTH
