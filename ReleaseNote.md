# Quick Start Guide

## Clustering 폴더 
- 'Edit_Distance_Hierarchical_Clustering.ipynb' 파일을 이용해 군집화를 진행할 수 있다. 
- 해당 폴더의 '5min_seq_trimmed_0301.csv'와 'places_info_id_0224.csv' 파일을 사용하여 문자열 간의 유사도를 계산한다. 
-  이 두 csv 파일을 한 폴더에 넣어 ipynb 파일을 첫번째 셀부터 실행하면 유사도 계산, 군집화, 최적의 군집 개수를 찾기 위한 수치 분석이 차례대로 진행된다.
- 유사도 계산에 약 24시간이 소요되기에 계산이 완료된 파일 'a_rdv_수정.txt' 을 함께 첨부하고자 하였으나 용량 문제로 업로드하지 못했다. 따라서 .ipynb 파일의 결과값을 참고하거나 실행 시간에 여유를 두고 실행하는 것을 권장한다.
  
 
## LSTM 폴더
- 'seq2seq_initial_wholeSequence_clst1.ipynb', 'seq2seq_initial_wholeSequence_clst2.ipynb' 파일을 이용해 각 cluster1, cluster2에 대한 모델을 실행할 수 있다.
- 폴더의 '5min_seq_intVec_added_max_clst.csv' 파일을 한 폴더에 넣어 첫번째 셀부터 실행하면 모델 학습 후 모델 예측 결과를 얻을 수 있다.
 
