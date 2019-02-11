import spacy
import operator

nlp = spacy.load('en', disable=['parser', 'ner'])

# tag dictionary to memorize historical tag data
# structure has [token] [history tags]
# in history tags, its store all history tags and associate counts
# always return the tag with max count.

tag_dictionary_1000292033 = {}


# a helper function to correct the spacy tag
def tag(body):
    # format data
    body = ' '.join(body.split())
    utt = nlp(body)

    result = []
    for token in utt:
        tag = new_tag(token.text, token.tag_)
        result.append(token.text + '/' + tag)

    result = ' '.join(result)

    return result


def new_tag(token, tag):
    # if the token is seen before
    if token in tag_dictionary_1000292033:
        history_tags = tag_dictionary_1000292033[token]

        # if the tag is seen before
        if tag in history_tags:
            history_tags[tag] = history_tags[tag] + 1
        # insert a new tag
        else:
            history_tags[tag] = 1

    else:
        new_token = {}
        new_token[tag] = 1
        tag_dictionary_1000292033[token] = new_token

    return max(tag_dictionary_1000292033[token].items(), key=operator.itemgetter(1))[0]