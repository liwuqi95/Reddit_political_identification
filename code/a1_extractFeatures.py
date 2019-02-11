import numpy as np
import argparse
import json
import re
import string
import csv

featsdir_1000292033 = '/u/cs401/A1/feats/'
listdir_1000292033 = '/u/cs401/Wordlists/'

# listdir_1000292033 = '../wordlists/'
# featsdir_1000292033 = '../feats/'

# preload the files
first_person_list_1000292033 = open(listdir_1000292033 + 'First-person').read().splitlines()
second_person_list_1000292033 = open(listdir_1000292033 + 'Second-person').read().splitlines()
third_person_list_1000292033 = open(listdir_1000292033 + 'Third-person').read().splitlines()
slang_list_1000292033 = open(listdir_1000292033 + 'Slang').read().splitlines()

bgl_list_1000292033 = {}
bgl_data_1000292033 = csv.reader(open(listdir_1000292033 + 'BristolNorms+GilhoolyLogie.csv'), delimiter=',')
next(bgl_data_1000292033)

for row in bgl_data_1000292033:
    bgl_list_1000292033[row[1]] = row

ra_list_1000292033 = {}
ra_data_1000292033 = csv.reader(open(listdir_1000292033 + 'Ratings_Warriner_et_al.csv'), delimiter=',')
next(ra_data_1000292033)

for row in ra_data_1000292033:
    ra_list_1000292033[row[1]] = row


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    # set placeholder
    features = np.zeros(173)
    # remove extra space
    comment = ' '.join(comment.split())
    # add a leading space for corner cases
    comment = ' ' + comment

    for first_person in first_person_list_1000292033:
        features[0] = features[0] + comment.count(' ' + first_person.lower() + '/')
    for second_person in second_person_list_1000292033:
        features[1] = features[1] + comment.count(' ' + second_person.lower() + '/')
    for third_person in third_person_list_1000292033:
        features[2] = features[2] + comment.count(' ' + third_person.lower() + '/')

    features[3] = comment.count('/CC ')
    features[4] = comment.count('/VBD ')
    features[5] = len(re.findall("'ll|will|gonna|(going/VBG\sto/TO\s\w*/VB)", comment))
    features[6] = comment.count(' ,')
    # two cases.
    # one, the punctuations are already splits, e.g.  !?! => !/. ?/. !/.
    # two, the punctuations are in normal, e.g. ...
    features[7] = len(
        re.findall("(\s[" + string.punctuation + "]/[.])(\s[" + string.punctuation + "]/[.]+)+", comment)) + len(
        re.findall("\s[" + string.punctuation + "][" + string.punctuation + "]+/", comment))
    features[8] = comment.count('/NN ') + comment.count('/NNS ')
    features[9] = comment.count('/NNP ') + comment.count('/NNPS ')
    features[10] = comment.count('/RB ')
    features[11] = comment.count('/WP ') + comment.count('/WP$ ') + comment.count('/WRB ')

    for slang in slang_list_1000292033:
        features[12] = features[12] + comment.count(' ' + slang.lower() + '/')

    features[13] = len(re.findall("\s[A-Z][A-Z][A-Z]+/", comment))
    features[14] = np.mean(list(map(lambda sentence: len(sentence.strip().split()), comment.split('\n'))))

    token_list = re.sub("\s[" + string.punctuation + "]*/.\s", " ", comment).split()

    length = 0
    for i in range(len(token_list)):
        length += len((token_list[i].rsplit('/', 1))[0])
    features[15] = (length / len(token_list)) if len(token_list) > 0 else 0

    features[16] = len(comment.split('\n'))

    token_list = comment.replace('\n', '').split()

    aoa_list = []
    img_list = []
    fam_list = []

    vms_list = []
    ams_list = []
    dms_list = []

    # extract data from the files data
    for token in token_list:
        token_info = token.rsplit('/', 1)[0]
        if token_info != '' and token_info in bgl_list_1000292033:
            aoa_list.append(float(bgl_list_1000292033[token_info][3]))
            img_list.append(float(bgl_list_1000292033[token_info][4]))
            fam_list.append(float(bgl_list_1000292033[token_info][5]))
        else:
            aoa_list.append(0.0)
            img_list.append(0.0)
            fam_list.append(0.0)
        if token_info != '' and token_info in ra_list_1000292033:
            vms_list.append(float(ra_list_1000292033[token_info][2]))
            ams_list.append(float(ra_list_1000292033[token_info][5]))
            dms_list.append(float(ra_list_1000292033[token_info][8]))
        else:
            vms_list.append(0.0)
            ams_list.append(0.0)
            dms_list.append(0.0)

    features[17] = np.average(aoa_list)
    features[18] = np.average(img_list)
    features[19] = np.average(fam_list)

    features[20] = np.std(aoa_list)
    features[21] = np.std(img_list)
    features[22] = np.std(fam_list)

    features[23] = np.average(vms_list)
    features[24] = np.average(ams_list)
    features[25] = np.average(dms_list)

    features[26] = np.std(vms_list)
    features[27] = np.std(ams_list)
    features[28] = np.std(dms_list)

    return features


def extract2(id, cat):
    ids = open(featsdir_1000292033 + cat + '_IDs.txt').read().splitlines()
    for i in range(len(ids)):
        if id == ids[i]:
            cat_feats = np.load(featsdir_1000292033 + cat + '_feats.dat.npy')
            return cat_feats[i]

    return np.zeros(144)


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173 + 1))
    cats = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

    for i in range(len(data)):
        feats[i][:29] = extract1(data[i]["body"])[:29]
        feats[i][29: 173] = extract2(data[i]["id"], data[i]["cat"])
        feats[i][173] = cats[data[i]["cat"]]

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    main(args)
