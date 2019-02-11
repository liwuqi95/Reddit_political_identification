import sys
import argparse
import os
import json
import re
import html
import string
import spacy

indir_1000292033 = '/u/cs401/A1/data/'
listdir_1000292033 = '/u/cs401/Wordlists/'

# indir_1000292033 = '../data/'
# listdir_1000292033 = '../wordlists/'


abbrev_list_1000292033 = open(listdir_1000292033 + 'abbrev.english').read().splitlines()
abbrev_list_space_1000292033 = list(map(lambda x: x.replace('.', ' .'), abbrev_list_1000292033))

clitic_list_1000292033 = open(listdir_1000292033 + 'clitics').read().splitlines()
clitic_list_space_1000292033 = list(map(lambda x: x.replace('\'', ' \' '), clitic_list_1000292033))

stop_list_1000292033 = open(listdir_1000292033 + 'StopWords').read().splitlines()

nlp = spacy.load('en', disable=['parser', 'ner'])


def preproc1(comment, steps=range(1, 11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    # set the modComm
    modComm = comment
    if 1 in steps:
        # remove new line
        modComm = modComm.replace('\n', ' ')
    if 2 in steps:
        # use html to remove html special characters
        modComm = html.unescape(modComm)
    if 3 in steps:
        # remove http or www url ending with space
        modComm = re.sub("(http|www).*\s", " ", modComm)
    if 4 in steps:
        # insert space aside the group of punctuations
        modComm = re.sub(r"([" + string.punctuation + "]+)", r" \1 ", modComm)
        # set Apostrophes back normal
        modComm = modComm.replace(" ...", "...")
        # set abbreviations back normal
        for i in range(len(abbrev_list_space_1000292033)):
            modComm = modComm.replace(abbrev_list_space_1000292033[i], abbrev_list_1000292033[i])
    if 5 in steps:
        # consider two cases.
        # 1. the punctuations are already splited with a space
        # 2. the punctuations are in normal condition
        for i in range(len(clitic_list_1000292033)):
            modComm = modComm.replace(clitic_list_space_1000292033[i], clitic_list_1000292033[i])
            modComm = modComm.replace(clitic_list_1000292033[i], ' ' + clitic_list_1000292033[i] + ' ')
    if 6 in steps:
        # remove extra spaces
        modComm = ' '.join(modComm.split())
        utt = nlp(modComm)
        token_list = []

        # set token in required format
        for token in utt:
            token_list.append(token.text + '/' + token.tag_)
        modComm = ' '.join(token_list)
    if 7 in steps:
        # remove stop words
        for stop_word in stop_list_1000292033:
            modComm = re.sub("\s" + stop_word + "/\w*\s", " ", modComm)
    if 8 in steps:
        # replace token text with lemma
        for token in utt:
            if token.text != token.lemma_:
                if not (token.lemma_.startswith('-') and not token.text.startswith('-')):
                    modComm = modComm.replace(token.text, token.lemma_)
    if 9 in steps:
        # insert new line after group of !?.
        modComm = re.sub(r"([!?.]/[.]\s)(\w)", r"\1 \n \2", modComm)
    if 10 in steps:
        token_list = modComm.split(" ")
        for i in range(len(token_list)):
            # check if the spacy tag applied
            token_info = token_list[i].rsplit('/', 1)
            if len(token_info) > 1:
                # case one, after spacy
                token_list[i] = token_info[0].lower() + '/' + token_info[1]
            else:
                # case two, raw data
                token_list[i] = token_list[i].lower()

        modComm = " ".join(token_list)

    # remove extra space
    modComm = re.sub(r" +", " ", modComm)
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir_1000292033):
        for file in files:
            fullFile = os.path.join(subdir, file)

            data = json.load(open(fullFile))

            count = args.max
            # set index with student number
            i = args.ID[0] % len(data)

            while count > 0:
                j = json.loads(data[i])
                line = {"id": j["id"], "cat": file, 'body': preproc1(j["body"])}
                allOutput.append(line)
                # circular index
                i = i + 1 if i < len(data) - 1 else 0
                count = count - 1

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", type=int, default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
