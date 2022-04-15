# Modified DeepMine Path
__DSPATH__ = "./DeepMine"

# output path
__OUTPATH__ = "./Data"

import os
from scipy.io import wavfile
from pydub import AudioSegment
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import shutil

def split(sound):
    dBFS = sound.dBFS
    chunks = split_on_silence(sound,
        min_silence_len = 100,
        silence_thresh = dBFS-16,
        keep_silence = 100
    )
    return chunks

def combine(_src):
    audio = AudioSegment.empty()
    for i,filename in enumerate(os.listdir(_src)):
        if filename.endswith('.wav'):
            filename = os.path.join(_src, filename)
            audio += AudioSegment.from_wav(filename)
    return audio


def save_chunks(chunks, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    counter = 0

    target_length = 5 * 1000
    output_chunks = [chunks[0]]
    lencounter = 0
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # if the last output chunk is longer than the target length,
            # we can start a new one
            output_chunks.append(chunk)
            lencounter+=1
        if(lencounter>120):
            break

    for chunk in output_chunks:
        chunk = chunk.set_frame_rate(24000)
        chunk = chunk.set_channels(1)
        counter = counter + 1
        chunk.export(os.path.join(directory, str(counter) + '.wav'), format="wav")

rawdspath = input("please enter full path of dataset (e.g : /home/drzeinali/Desktop/wav/")

file2speaker = open("file2speaker.lst")
lines = file2speaker.readlines()
file2speakerDic = {}

for line in lines:
    data = line.strip().split(' ')
    file2speakerDic[data[0]]= data[1]

if not os.path.exists(__DSPATH__):    
    os.makedirs(__DSPATH__)
if not os.path.exists(__OUTPATH__):
    os.makedirs(__OUTPATH__)    

filelist = os.listdir(rawdspath)
for file in filelist:
    fileid = file.split('.')[0]
    speakerid = file2speakerDic[fileid]
    src = rawdspath + file
    path = __DSPATH__+ '/p' +speakerid
    if(not os.path.exists(path)):
        os.makedirs(path)
    shutil.copy(src, path)


speakers = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040','041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','066']

for p in speakers:
    directory = __OUTPATH__ + '/p' + str(p)
    if not os.path.exists(directory):
        audio = combine(__DSPATH__ + '/p' + str(p))
        chunks = split(audio)
        save_chunks(chunks, directory)




data_list = []
for path, subdirs, files in os.walk(__OUTPATH__):
    for name in files:
        if name.endswith(".wav"):
            speaker = path.split('/')[-1].replace('p', '')
            if speaker in speakers:
                data_list.append({"Path": os.path.join(path, name), "Speaker": int(speakers.index(speaker)) + 1})
                

data_list = pd.DataFrame(data_list)
data_list = data_list.sample(frac=1)


split_idx = round(len(data_list) * 0.1)

test_data = data_list[:split_idx]
train_data = data_list[split_idx:]



file_str = ""
for index, k in train_data.iterrows():
    file_str += k['Path'] + "|" +str(k['Speaker'] - 1)+ '\n'
text_file = open(__OUTPATH__ + "/train_list.txt", "w")
text_file.write(file_str)
text_file.close()

file_str = ""
for index, k in test_data.iterrows():
    file_str += k['Path'] + "|" + str(k['Speaker'] - 1) + '\n'
text_file = open(__OUTPATH__ + "/val_list.txt", "w")
text_file.write(file_str)
text_file.close()