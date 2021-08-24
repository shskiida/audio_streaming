#ストリーミング加工
#音声の周波数成分可視化(カットオフ2)
#FFT
#周波数カットオフ
#逆FFT
#保存

import requests
import sys
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import wave
import os
from scipy.io.wavfile import write
import pyaudio
import datetime

#録音する時間の長さ（秒）
recsec = 2
iDeviceIndex = 1 #録音デバイスのインデックス番号
 
FORMAT = pyaudio.paInt16 #音声のフォーマット
CHANNELS = 1             #モノラル
RATE = 44100             #サンプルレート
CHUNK = 2**12         #データ点数
audio = pyaudio.PyAudio() #pyaudio.PyAudio()
rate = 44100

fcmaxU = 2200 #上限カットオフ周波数
fcmaxL = -fcmaxU 
fcminU = 100 #下限カットオフ周波数
fcminL = -fcminU
fc1 = 135
fc2 = 137
    
while True:
    if os.path.exists('data.csv') == False:
        
        dataset = ['Time' , 'Max' , 'Ave' , 'score\n']
        
        
        with open('data.csv','w') as f:
            f.writelines(' , '.join(dataset))
        
    
    
    
    nowtime = datetime.datetime.now().strftime('%Y/%m/%d_%H:%M:%S')
    stream = audio.open(format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True, output=True,
        input_device_index = iDeviceIndex, 
        frames_per_buffer=CHUNK)
    
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * recsec)):
        rawdata = stream.read(CHUNK)
        frames.append(rawdata)
    stream.close()
    a = np.array(frames)    

    ndarray = np.frombuffer(a, dtype='int16')

#（振幅）の配列を作成
    data = ndarray / 32768
    time = np.arange(0, data.shape[0]/rate, 1/rate)
#データプロット
    #plt.plot(time, data)
    #plt.ylim(-1.0,1.0)
    #plt.xlabel('time(s)')
    #plt.ylabel('amplitude')
    #plt.title('streaming_wave')
    #plt.show()


#FFTにより周波数成分を表示する。
#縦軸：dataを高速フーリエ変換する（時間領域から周波数領域に変換する）
    F = np.fft.fft(data)
    fft_data = np.abs(F)

#横軸：周波数の取得　　#np.fft.fftfreq(データ点数, サンプリング周期)
    freqList = np.fft.fftfreq(data.shape[0], d=1.0/rate)

#データプロット
    #plt.plot(freqList, fft_data)
    #plt.ylim(0,2000)
    #plt.xlim(0,6000) #0～6000Hzまで表示
    #plt.xlabel('Frequency[Hz]')
    #plt.ylabel('amplitude')
    #plt.title('streaming_wave')
    #plt.show()

#周波数指定してフィルタリング
    F_cut = np.copy(F)
    F_cut[(freqList > fcmaxU) | (freqList < fcmaxL)] = 0 #カットオフ周波数のデータを0にする。
    F_cut[(freqList > fcminL) & (freqList < fcminU)] = 0 #カットオフ周波数のデータを0にする。
    F_cut[(freqList > fc1) & (freqList < fc2)] = 0 #カットオフ周波数以下のデータを0にする。

    fft_data_cut = np.abs(F_cut)


#max,aveの算出
    max_data = round(max(fft_data_cut),1)
    nonzerodata = np.count_nonzero(fft_data_cut) #fft_data_cutの中から0ではないデータ個数のカウント
    ave_data = round(sum(fft_data_cut)/nonzerodata,1) #平均値の算出

#振幅によってスコアを決める。
    if ave_data < 5 :
        score = 0
    elif ave_data >= 5 and ave_data < 10 :
        score = 1
    elif ave_data >= 10 and ave_data < 15 :
        score = 2
    elif ave_data >= 15 and ave_data < 20 :
        score = 3
    elif ave_data >= 20 and ave_data < 25 :
        score = 4
    elif ave_data >=25 :
        score = 5


    print(nowtime)
    print(f'max={max_data}')
    print(f'ave={ave_data}')
    print(f'score={score}')
    
    data_arr = []
    data_arr.append(nowtime)
    data_arr.append(f'{max_data}')
    data_arr.append(f'{ave_data}')
    data_arr.append(f'{score}\n')
    
#データプロット
    #plt.plot(freqList, fft_data_cut)
    #plt.ylim(0,2000)
    #plt.xlim(0,6000) #0～6000Hzまで表示
    #plt.xlabel('Frequency[Hz]')
    #plt.ylabel('amplitude')
    #plt.title('streaming_wave')
    #plt.show()

#逆FFT
    #newF = np.fft.ifft(F_cut)

    #time = np.arange(0, newF.shape[0]/rate, 1/rate)  
#データプロット
    #plt.plot(time, newF)
    #plt.ylim(-1.0,1.0)
    #plt.xlabel('time(s)')
    #plt.ylabel('amplitude')
    #plt.title('streaming_wave')
    #plt.show()


#処理後データの保存
    #newF = newF*32768
    #newFi = newF.real.astype(np.int16)
    #write('sample.wav', rate, newFi)
    #print("処理後のデータを保存しました。")
    
    with open ('data.csv','a') as f:
        f.writelines(' , '.join(data_arr))
        
