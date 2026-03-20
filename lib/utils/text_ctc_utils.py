import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    sign_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        sign_preds.append(remove_duplicates(tp))
    return sign_preds

def numerize(sents, vocab_map,full_transformer):
    outs = []
    for sent in sents:
        if type(sent) != float :
            if full_transformer:
                outs.append([32]+ list(map(lambda x: vocab_map[x], sent))+ [0])
            else:
                outs.append(list(map(lambda x: vocab_map[x], sent)))

    return outs

def invert_to_chars(sents, inv_ctc_map):
    sents = sents.detach().numpy()
    outs = []
    for sent in sents:
        for x in sent:
            if x == 0:
                break
            outs.append(inv_ctc_map[x]) 
    return outs

def get_ctc_vocab(char_list):
    # blank
    ctc_char_list = "_" + char_list
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(ctc_char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, ctc_char_list

def get_autoreg_vocab(char_list):
    # blank
    ctc_map, inv_ctc_map = {}, {}
    for i, char in enumerate(char_list):
        ctc_map[char] = i
        inv_ctc_map[i] = char
    return ctc_map, inv_ctc_map, char_list


def convert_text_for_ctc(DATASET_CSV_PATH,vocab_map,full_transformer=False):
    all_data = pd.read_csv(DATASET_CSV_PATH)
    all_data = all_data[all_data['filename'].notna()]
    all_data = all_data[all_data['label_proc'].notna()]
    label = all_data["label_proc"]
    
    targets_enc = numerize(label, vocab_map,full_transformer)

    # targets = [[c for c in x] for x in label]
    # targets_flat = [c for clist in targets for c in clist]
    # lbl_enc = preprocessing.LabelEncoder()
    # lbl_enc.fit(targets_flat)
    # targets_enc = [lbl_enc.transform(x) for x in targets]
    # targets_enc = np.array(targets_enc)
    # targets_enc = targets_enc + 1
    
    df = pd.DataFrame()
    df["names"] = all_data["filename"]
    df["enc"] = targets_enc

    # print("number of classes after conversion for CTC", lbl_enc.classes_)
    
    return  df

    # return  df , lbl_enc
