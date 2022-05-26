#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import sys
from tensorflow import keras
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import string
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
import unicodedata
import urllib3
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# In[3]:


from ast import literal_eval
df = pd.read_csv('5min_seq_intVec_added_max_clst.csv')
condition = (df.cluster == 1)
df_clst2_100 = df[condition]
df_clst2_100


# In[4]:


# from tensorflow.keras.preprocessing.sequence import pad_sequences
# train_seq = 
avg = df_clst2_100['length_vec'].mean()
med = df_clst2_100['length_vec'].median()
print(avg)
print(med)


# In[5]:


from ast import literal_eval
df_clst2_100['seq_int_vec'] = df_clst2_100['seq_int_vec'].apply(lambda x: literal_eval(str(x)))


# In[6]:


all_data = df_clst2_100['seq_int_vec']
all_data = all_data.values
all_data


# In[7]:


seq =  df_clst2_100.iloc[0]['sequence']
integer = df_clst2_100.iloc[0]['seq_int_vec']
print('Sequence: ',seq)
print(" ")
print('Mapped integers: ',integer)


# In[8]:


all_len = []
for i in all_data:
    all_len.append(len(i))

all_len = np.array(all_len)


# In[9]:


print('길이 최솟값: ' ,min(all_len))
print('길이 최댓값: ' ,max(all_len))
print('길이 중간값: ' , np.mean(all_len))
print('길이 평균값: ' , np.median(all_len))


# In[10]:


from collections import OrderedDict

total_input = []
total_target = []
input_without_repeat = []
target_without_repeat = []

time_steps = 15 # input data의 time steps
for_periods = 10 # output data의 time steps

for i in range(len(all_data)):  
    seq_len = len(all_data[i])
    if seq_len < 25:
        print(i, seq_len)
        continue
    for j in range(time_steps, seq_len-1):   
        ori_input = all_data[i][j-time_steps:j]
        ori_target = all_data[i][j:j+for_periods]
        if(len(ori_input)>=time_steps and len(ori_target)>=for_periods):
            total_input.append(ori_input)
            total_target.append(ori_target)
        else: continue

for x in total_input:
    input_without_repeat.append(list(OrderedDict.fromkeys(x)))

for y in total_target:
    target_without_repeat.append(list(OrderedDict.fromkeys(y)))


# In[11]:


input_without_repeat


# In[12]:


target_without_repeat


# In[13]:


import copy

encoder_input = copy.deepcopy(total_input)
encoder_input = np.array(encoder_input)


# In[14]:


decoder_input = copy.deepcopy(total_target)

for i in decoder_input:
    i.insert(0, 2)
    
decoder_input = np.array(decoder_input)
print(decoder_input)


# In[15]:


decoder_target = copy.deepcopy(total_target)

for i in decoder_target:
    length = len(i)
    i.insert(len(i), 3)

decoder_target = np.array(decoder_target)
print(decoder_target)


# In[16]:


print('인코더의 입력의 크기(shape) :',encoder_input.shape)
print('디코더의 입력의 크기(shape) :',decoder_input.shape)
print('디코더의 레이블의 크기(shape) :',decoder_target.shape)


# In[17]:


print(len(encoder_input))
print(len(decoder_input))
print(len(decoder_target))


# In[18]:


indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
print('랜덤 시퀀스 :',indices)


# In[19]:


encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]


# In[20]:


encoder_input[97]


# In[21]:


decoder_input[97]


# In[22]:


decoder_target[97]


# In[23]:


n_of_val = int(len(encoder_input)*0.1)
print('검증 데이터의 개수 :',n_of_val)


# In[24]:


n_of_test = int(len(encoder_input)*0.1)
print('테스트 데이터의 개수 :',n_of_test)


# In[25]:


encoder_input_train = encoder_input[0:len(encoder_input)-n_of_val-n_of_test]
decoder_input_train = decoder_input[0:len(decoder_input)-n_of_val-n_of_test]
decoder_target_train = decoder_target[0:len(decoder_target)-n_of_val-n_of_test]

encoder_input_val = encoder_input[len(encoder_input)-n_of_val-n_of_test:len(encoder_input)-n_of_val]
decoder_input_val = decoder_input[len(decoder_input)-n_of_val-n_of_test:len(decoder_input)-n_of_val]
decoder_target_val = decoder_target[len(decoder_target)-n_of_val-n_of_test:len(decoder_target)-n_of_val]

encoder_input_test = encoder_input[len(encoder_input)-n_of_val:]
decoder_input_test = decoder_input[len(decoder_input)-n_of_val:]
decoder_target_test = decoder_target[len(decoder_target)-n_of_val:]


# In[26]:


print('training source 데이터의 크기 :',encoder_input_train.shape)
print('training target 데이터의 크기 :',decoder_input_train.shape)
print('training target 레이블의 크기 :',decoder_target_train.shape)

print('validation source 데이터의 크기 :',encoder_input_val.shape)
print('validation target 데이터의 크기 :',decoder_input_val.shape)
print('validation target 레이블의 크기 :',decoder_target_val.shape)

print('test source 데이터의 크기 :',encoder_input_test.shape)
print('test target 데이터의 크기 :',decoder_input_test.shape)


# In[27]:


from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking, TimeDistributed
from tensorflow.keras.models import Model


# In[28]:


embedding_dim = 64
hidden_units = 64


# In[29]:


vocab_size = 7874


# In[30]:


# 인코더
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs) # 임베딩 층
# enc_masking = Masking(mask_value=0.0)(enc_emb) # 패딩 0은 연산에서 제외
encoder_lstm = LSTM(hidden_units, return_state=True) # 상태값 리턴을 위해 return_state는 True
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb) # 은닉 상태와 셀 상태를 리턴
encoder_states = [state_h, state_c] # 인코더의 은닉 상태와 셀 상태를 저장


# In[31]:


# 디코더
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, hidden_units) # 임베딩 층
dec_emb = dec_emb_layer(decoder_inputs) # 패딩 0은 연산에서 제외
# dec_masking = Masking(mask_value=0.0)(dec_emb)

# 상태값 리턴을 위해 return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True) 

# 인코더의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)

# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_softmax_outputs = decoder_dense(decoder_outputs)

# 모델의 입력과 출력을 정의.
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# In[32]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping_cb = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)

history = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,           validation_data=([encoder_input_val, decoder_input_val], decoder_target_val),
          batch_size=64, epochs=200, callbacks=[early_stopping_cb])

model.summary()


# In[33]:


from matplotlib import pyplot as plt

# # 정확도 시각화
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
# plt.show()

# loss 시각화
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
plt.show()


# In[34]:


# 인코딩 결과로 발생할 상태값도 가져오기 위해 그를 반환할 모델 (encoder_model)
encoder_model = Model(encoder_inputs, encoder_states)

encoder_h_state = Input(shape=(hidden_units,))
encoder_c_state = Input(shape=(hidden_units,))

# 디코더 설계 시작
# 이전 시점의 상태를 보관할 텐서
decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 훈련 때 사용했던 임베딩 층을 재사용
dec_emb2 = dec_emb_layer(decoder_inputs)

# 다음 단어 예측을 위해 이전 시점의 상태를 현 시점의 초기 상태로 사용
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# 모든 시점에 대해서 단어 예측
decoder_outputs2 = decoder_dense(decoder_outputs2)

# 수정된 디코더
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


# In[35]:


def decode_sequence(input_seq):
  # 입력으로부터 인코더의 마지막 시점의 상태(은닉 상태, 셀 상태)를 얻음
    states_value = encoder_model.predict(input_seq)

  # <SOS>에 해당하는 정수 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = 2

    stop_condition = False
    decoded_sentence = []

  # stop_condition이 True가 될 때까지 루프 반복
  # 구현의 간소화를 위해서 이 함수는 배치 크기를 1로 가정합니다.
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_words, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 단어로 변환
        predicted_word = inverse_places[np.argmax(output_words[0,0])]
        # print(predicted_word)

        # 현재 시점의 예측 단어를 예측 문장에 추가
        decoded_sentence.append(predicted_word)

        # <eos>에 도달하거나 정해진 길이를 넘으면 중단.
        if (predicted_word == 'end' or
            len(decoded_sentence) >= 10):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        predicted_seq = np.zeros((1,1))
        predicted_seq[0, 0] = np.argmax(output_words[0,0])

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence


# In[36]:


# !pip install import_ipynb 


# In[37]:


import import_ipynb
import seq_into_int_vectors as seqvec


# In[38]:


print(seqvec.places_dict)


# In[39]:


inverse_places = {v: k for k, v in seqvec.places_dict.items()}
print(inverse_places)


# In[40]:


# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq_to_src(input_seq):
    sentence = []
    for encoded_word in input_seq:
        if(encoded_word != 0):
            sentence.append(inverse_places[encoded_word])
    return sentence

# 결과의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq_to_tar(input_seq):
    sentence = []
    for encoded_word in input_seq:
        if(encoded_word != 0 and encoded_word != seqvec.places_dict['start'] and encoded_word != seqvec.places_dict['end']):
            sentence.append(inverse_places[encoded_word])
    return sentence


# In[41]:


from collections import OrderedDict

encoder_input_less_repeat = []
decoder_input_less_repeat = []

for i in range(len(decoder_input_test)):
    trimmed = list(OrderedDict.fromkeys(decoder_input_test[i]))
    if len(trimmed) >= 3:
        decoder_input_less_repeat.append(i)   

print(decoder_input_less_repeat)
print(len(decoder_input_less_repeat))


# In[42]:


len(encoder_input_test)


# In[43]:


def edit_dist(arr1, arr2):
    dp = [[0] * (len(arr2)+1) for _ in range(len(arr1) + 1)]
    for i in range(1, len(arr1)+1):
        dp[i][0] = i
    for j in range(1, len(arr2)+1):
        dp[0][j] = j

    for i in range(1, len(arr1)+1):
        for j in range(1, len(arr2)+1):
            if arr1[i-1] == arr2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

    return dp[-1][-1]


# In[44]:


# Mean Relatvie Error
def MRE(predicted_seq, target_seq):
    editDist = edit_dist(predicted_seq, target_seq)
    longerSeq = max(len(predicted_seq), len(target_seq))
    return editDist / longerSeq


# In[45]:


def sequence_trim(temp_list):
    # trimmed list 생성하는 부분
    pointer = temp_list[0]
    
    cnt_list = []
    final_list = []
    final_list.append(pointer)
    cnt = 1
    
    for j in range(1, len(temp_list) + 1):
        # 비교해서 같으면 패스, 다르면 리스트에 넣음
        if j == len(temp_list):
            cnt_list.append(cnt)    
            break
        if pointer == temp_list[j]: 
            cnt += 1
            continue
        else:
            pointer = temp_list[j]
            final_list.append(pointer)
            cnt_list.append(cnt)
            cnt = 1
    return final_list


# In[46]:


# Mean Relatvie Error

def MRE_trimmed(predicted_seq, target_seq):
    predicted_trimmed = sequence_trim(predicted_seq)
    target_trimmed = sequence_trim(target_seq)        
    editDist = edit_dist(predicted_trimmed, target_trimmed)
    longerSeq = max(len(predicted_seq), len(target_seq))
    return editDist / longerSeq


# In[47]:


# Mean accuracy
def MA(predicted_seq, target_seq):
    set1 = set(predicted_seq)
    set2 = set(target_seq)
    correctPredictNum = len(set1 & set2)
    
    return correctPredictNum / len(set1)


# In[48]:


mre_list = []
mre_trimmed_list = []
ma_list = []
target_trimmed = []
output_trimmed = []
test_results = []

good_results = []

seq_indices =  decoder_input_less_repeat

for seq_index in seq_indices:
    input_seq = encoder_input_test[seq_index: seq_index + 1]
    print(input_seq.shape)
    decoded_sentence = decode_sequence(input_seq)
    
    inputstr = ["입력 :"] + list(seq_to_src(encoder_input_test[seq_index]))
    target_trimmed.append(len(sequence_trim(list(seq_to_tar(decoder_input_test[seq_index])))))
    output_trimmed.append(len(sequence_trim(list(decoded_sentence)))) 
    targetstr = ["정답 :"] + list(seq_to_tar(decoder_input_test[seq_index]))
    outputstr = ["출력결과 :"] + list(decoded_sentence)
    test_results.append([inputstr, targetstr, outputstr])
    
    if(len(set(decoded_sentence)) >= 3 and len(set(seq_to_tar(decoder_input_test[seq_index]))) >= 3):
        good_results.append([inputstr, targetstr, outputstr])

    print(inputstr)
    print(targetstr)
    print(outputstr)
    mre_list.append(MRE(decoded_sentence, seq_to_tar(decoder_input_test[seq_index])))
    mre_trimmed_list.append(MRE_trimmed(decoded_sentence, seq_to_tar(decoder_input_test[seq_index])))
    ma_list.append(MA(decoded_sentence, seq_to_tar(decoder_input_test[seq_index])))
    print("-"*50)


# In[49]:


# 문자열 시퀀스를 정수 시퀀스로 바꿈
def seq_to_vec(input_seq):
    sequence = []
    for loc in input_seq:
        sequence.append(seqvec.places_dict[loc])
    return sequence


# In[50]:


# target trimmed
print("target trimmed 최댓값 :" , max(target_trimmed))
print("target trimmed 최솟값 :" , min(target_trimmed))
print("target trimmed 평균 :" , np.mean(np.array(target_trimmed)))
print("target trimmed 표준편차 :" , np.std(np.array(target_trimmed)))


# In[51]:


output_trimmed


# In[52]:


# 초기 경로 데이터
import math 
import time 
start = time.time() 


input_seq_str = np.array(['-', '-', '-', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114',             'Sx0049', 'Sx0049', 'Sx0049', 'Sx0049'])
input_seq = seq_to_vec(input_seq_str)
input_seq = np.reshape(input_seq, (1, 15))

# 정답 (실제 적용에서는 정답이 없으나 이 경우 정확도 비교를 위해 정답 가져옴)
target_seq_str = np.array(['Sx0049', 'Sx0049', 'Sx0049', '-', '-', '-', '-', '-', '-', '-'])
target_seq = seq_to_vec(target_seq_str)
target_seq = np.reshape(target_seq, (1, 10))

# 모델에 넣어 이후 경로 추천 (예측)
decoded_sentence = decode_sequence(input_seq)
end = time.time()

inputstr = "입력 :" + str(input_seq_str)
targetstr = "정답 :" + str(target_seq_str)

outputstr = "출력결과 :" + str(decoded_sentence)

print(inputstr)
print(targetstr)
print(outputstr)
print("-"*50)
 
print(f"{end - start:.5f} sec")


# In[53]:


for i in good_results:
    print(i)
    print(" ")


# In[54]:


ma_list


# In[55]:


mre_trimmed_list


# In[56]:


# MRE trimmed
print("MRE trimmed 최댓값 :" , max(mre_trimmed_list))
print("MRE trimmed 최솟값 :" , min(mre_trimmed_list))
print("MRE trimmed 평균 :" , np.mean(np.array(mre_trimmed_list)))
print("MRE trimmed 표준편차 :" , np.std(np.array(mre_trimmed_list)))


# In[57]:


values = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
c0 = mre_trimmed_list.count(0.0) / len(mre_trimmed_list)
c1 = mre_trimmed_list.count(0.1) / len(mre_trimmed_list)
c2 = mre_trimmed_list.count(0.2) / len(mre_trimmed_list)
c3 = mre_trimmed_list.count(0.3) / len(mre_trimmed_list)
c4 = mre_trimmed_list.count(0.4) / len(mre_trimmed_list)
c5 = mre_trimmed_list.count(0.5) / len(mre_trimmed_list)
c6 = mre_trimmed_list.count(0.6) / len(mre_trimmed_list)
c7 = mre_trimmed_list.count(0.7) / len(mre_trimmed_list)
c8 = mre_trimmed_list.count(0.8) / len(mre_trimmed_list)
c9 = mre_trimmed_list.count(0.9) / len(mre_trimmed_list)
c10 = mre_trimmed_list.count(1.0) / len(mre_trimmed_list)
counts = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
counts_cumul = np.cumsum(counts)
print(counts_cumul)


plt.bar(values, counts)
plt.title('MRE trimmed', fontsize = 14)
plt.xticks(values)
plt.yticks(fontsize = 14)
# plt.savefig('MRE_clst1.jpg')
plt.show()

plt.bar(values, counts_cumul, color='orange')
plt.title('MRE trimmed_Cumulative', fontsize = 14)
plt.xticks(values)
plt.yticks(fontsize = 14)
# plt.savefig('MRE_cumulative_clst1.jpg')
plt.show()


# In[58]:


# MRE trimmed 값이 평균 근처인 경우
for i in range(len(mre_trimmed_list)):
    if 0.0 <= mre_trimmed_list[i] <= 0.1:
        print(test_results[i][0])
        print(test_results[i][1])
        print(test_results[i][2])
        print("-"*50)


# In[59]:


df2=pd.read_csv('places_info_id_0224.csv',encoding='cp949')
tyid=df2['type_id']
tyid=list(tyid)
lat=df2['lat']
lat=list(lat)
lng=df2['lng']
lng=list(lng)


# In[60]:


get_ipython().system(' pip install haversine')


# In[61]:


from haversine import haversine


# In[62]:


# MRE trimmed 값이 최댓값에 가까운 경우
for i in range(len(mre_trimmed_list)):
    distances = []
    if 0.4 <= mre_trimmed_list[i] <= 0.6:
        print(test_results[i][0])
        print(test_results[i][1])
        print(test_results[i][2])
        for j in range(1, 11):
            type_id1=test_results[i][1][j]
            type_id2=test_results[i][2][j]
            if(type_id1 == '-' or type_id2 == '-'):
                distances.append("-")
                continue

            gps1=(lat[tyid.index(type_id1)],lng[tyid.index(type_id1)])
            gps2=(lat[tyid.index(type_id2)],lng[tyid.index(type_id2)])

            dist=haversine(gps1,gps2,unit='m')
            distances.append(dist)
        print(distances)
        print("-"*50)


# In[63]:


# MRE
print("MRE 최댓값 :" , max(mre_list))
print("MRE 최솟값 :" , min(mre_list))
print("MRE 평균 :" , np.mean(np.array(mre_list)))
print("MRE 표준편차 :" , np.std(np.array(mre_list)))


# In[64]:


# MRE 값이 좋은(작은) 경우
for i in range(len(mre_list)):
    distances = []
    if 0.0 <= mre_list[i] <= 0.2:
        print(test_results[i][0])
        print(test_results[i][1])
        print(test_results[i][2])
        for j in range(1, 11):
            type_id1=test_results[i][1][j]
            type_id2=test_results[i][2][j]
            if(type_id1 == '-' or type_id2 == '-'):
                distances.append('-')
                continue
            if(type_id1 == 'end' or type_id2 == 'end'):
                continue

            gps1=(lat[tyid.index(type_id1)],lng[tyid.index(type_id1)])
            gps2=(lat[tyid.index(type_id2)],lng[tyid.index(type_id2)])

            dist=haversine(gps1,gps2,unit='m')
            distances.append(dist)
        print(distances)
        print("-"*50)


# In[65]:


# MRE 값이 평균 근처인 경우
for i in range(len(mre_list)):
    distances = []
    if 0.4 <= mre_list[i] <= 0.6:
        print(test_results[i][0])
        print(test_results[i][1])
        print(test_results[i][2])
        
        if len(test_results[i][2]) < 11:
            continue
        
        for j in range(1, 11):
            type_id1=test_results[i][1][j]
            type_id2=test_results[i][2][j]
            if(type_id1 == '-' or type_id2 == '-'):
                distances.append('-')
                continue
            if(type_id1 == 'end' or type_id2 == 'end'):
                continue

            gps1=(lat[tyid.index(type_id1)],lng[tyid.index(type_id1)])
            gps2=(lat[tyid.index(type_id2)],lng[tyid.index(type_id2)])

            dist=haversine(gps1,gps2,unit='m')
            distances.append(dist)
        print(distances)
        print("-"*50)


# In[66]:


# MRE 값이 나쁜(큰) 경우
for i in range(len(mre_list)):
    distances = []
    if 0.8 <= mre_list[i] <= 1.0:
        print(test_results[i][0])
        print(test_results[i][1])
        print(test_results[i][2])
        for j in range(1, 11):
            type_id1=test_results[i][1][j]
            type_id2=test_results[i][2][j]
            if(type_id1 == '-' or type_id2 == '-'):
                distances.append('-')
                continue
            if(type_id1 == 'end' or type_id2 == 'end'):
                continue

            gps1=(lat[tyid.index(type_id1)],lng[tyid.index(type_id1)])
            gps2=(lat[tyid.index(type_id2)],lng[tyid.index(type_id2)])

            dist=haversine(gps1,gps2,unit='m')
            distances.append(dist)
        print(distances)
        print("-"*50)


# In[67]:


# MA
print("MA 최댓값 :" , max(ma_list))
print("MA 최솟값 :" , min(ma_list))
print("MA 평균 :" , np.mean(np.array(ma_list)))
print("MA 표준편차 :" , np.std(np.array(ma_list)))


# In[68]:


for i in range(len(ma_list)):
    if 0.8 <= ma_list[i] <= 1.0:
        print(test_results[i][0])
        print(test_results[i][1])
        print(test_results[i][2])
        print("-"*50)


# In[69]:


values = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
c0 = mre_list.count(0.0) / len(mre_list)
c1 = mre_list.count(0.1) / len(mre_list)
c2 = mre_list.count(0.2) / len(mre_list)
c3 = mre_list.count(0.3) / len(mre_list)
c4 = mre_list.count(0.4) / len(mre_list)
c5 = mre_list.count(0.5) / len(mre_list)
c6 = mre_list.count(0.6) / len(mre_list)
c7 = mre_list.count(0.7) / len(mre_list)
c8 = mre_list.count(0.8) / len(mre_list)
c9 = mre_list.count(0.9) / len(mre_list)
c10 = mre_list.count(1.0) / len(mre_list)
counts = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
counts_cumul = np.cumsum(counts)
print(counts)


plt.bar(values, counts)
plt.title('MRE', fontsize = 14)
plt.xticks(values)
plt.yticks(fontsize = 14)
# plt.savefig('MRE_clst1.jpg')
plt.show()

plt.bar(values, counts_cumul, color='orange')
plt.title('MRE_Cumulative', fontsize = 14)
plt.xticks(values)
plt.yticks(fontsize = 14)
# plt.savefig('MRE_cumulative_clst1.jpg')
plt.show()


# In[70]:


mre_list


# In[71]:


bins = np.arange(0, 1.2, 0.1)
hist, bins = np.histogram(np.array(ma_list), bins)

ma_values = ['[0.0, 0.1)', '[0.1, 0.2)', '[0.2, 0.3)', '[0.3, 0.4)', '[0.4, 0.5)', '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 0.9)', '[0.9, 1.0)', '1.0']
denominator = np.array([len(ma_list) for i in range(len(ma_values))])
print(hist)
print(denominator)
ma_counts = hist / len(ma_list)
print(ma_counts)
print(ma_values)
ma_counts_cumul = np.cumsum(ma_counts)

# plt.hist((ma_array), bins, rwidth = 0.8)
# plt.title('MA', fontsize = 14)
# plt.xticks(np.arange(0, 1.0, 0.1))
# plt.yticks(fontsize = 14)
# plt.show()

plt.bar(ma_values, ma_counts)
plt.title('MA', fontsize = 14)
plt.xticks(ma_values)
plt.xticks(rotation=45)
plt.yticks(fontsize = 14)
plt.savefig('MA_clst1.jpg')
plt.show()

plt.bar(ma_values, ma_counts_cumul, color='orange')
plt.title('MA_Cumulative', fontsize = 14)
plt.xticks(ma_values)
plt.xticks(rotation=45)
plt.yticks(fontsize = 14)
plt.savefig('MA_cumulative_clst1.jpg')
plt.show()


# In[72]:


len(ma_list)


# In[73]:


sum(hist)


# In[74]:


# 초기 경로 데이터
import math 
import time 
start = time.time() 


input_seq_str = np.array(['-', '-', '-', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114', 'Bs0114',             'Sx0049', 'Sx0049', 'Sx0049', 'Sx0049'])
input_seq = seq_to_vec(input_seq_str)
input_seq = np.reshape(input_seq, (1, 15))

# 정답 (실제 적용에서는 정답이 없으나 이 경우 정확도 비교를 위해 정답 가져옴)
target_seq_str = np.array(['Sx0049', 'Sx0049', 'Sx0049', '-', '-', '-', '-', '-', '-', '-'])
target_seq = seq_to_vec(target_seq_str)
target_seq = np.reshape(target_seq, (1, 10))

# 모델에 넣어 이후 경로 추천 (예측)
decoded_sentence = decode_sequence(input_seq)
end = time.time()

inputstr = "입력 :" + str(input_seq_str)
targetstr = "정답 :" + str(target_seq_str)

outputstr = "출력결과 :" + str(decoded_sentence)

print(inputstr)
print(targetstr)
print(outputstr)
print("-"*50)
 
print(f"{end - start:.5f} sec")


# In[ ]:




