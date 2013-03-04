import nltk
import random

#read training file, read grammar, try to parse each sentence, return number of sentences analyzed and number of analyses per sentence

def doNothing(words,tags):
    return words,tags

def preprocess(words,tags):
    for i, (word, tag) in enumerate(zip(words,tags)):
        if tag == 'P' and (word.lower() == 'to' or word == '2'):
            tags[i] = '2'
        if tag == 'V' and (word.lower() == 'can' or word == 'must' or word == 'may' or word == 'could' or word == 'would' or word =='get' or word == 'have'):
            tags[i] = 'MD'
        if tag == '~' and (word.lower() == 'rt'):
            tags[i] = 'RT'
        if tag == 'V' and (word.lower() == 'going'):
            tags[i] = 'GOING'
    return words,tags

def conllSeqGenerator(input_file):
    """ return an instance generator for a filename

    The generator yields lists of words and tags.  For test data, the tags
    may be unknown.  For usage, see trainClassifier and applyClassifier below.

    """
    cur_words = []
    cur_tags = []
    with open(input_file) as instances:
        for line in instances:
            if len(line.rstrip()) == 0:
                if len(cur_words) > 0:
                    yield cur_words,cur_tags
                    cur_words = []
                    cur_tags = []
            else:
                parts = line.rstrip().split()
                cur_words.append(parts[0])
                if len(parts)>1:
                    cur_tags.append(parts[1])
                else: cur_tags.append(unk)
        if len(cur_words)>0: 
            yield cur_words,cur_tags

def getShuffledTags(tags):
    out = list(tags) #copy elements
    random.shuffle(out)
    return out

def parseTags(tags,parser):
    trees = []
    try:
        trees = parser.nbest_parse(tags)
    except:
        pass
    return(trees)

def evalParser(cfg,input_file="oct27.dev",debug=False,max_len=20,preprocessor=doNothing,seed=0):
    random.seed(seed)
    grammar = nltk.data.load(cfg,cache=False)
    parser = nltk.ChartParser(grammar)
    tp = 0.0
    fp = 0.0
    fn = 0.0
    num_parses = 0.0
    unparsed = []
    for words,tags in conllSeqGenerator(input_file):
        words,tags = preprocessor(words,tags)
        
        # result = ""
        # for tag in tags:
        #     result += "\""+tag+"\""
        #     result += " "
        # print "S -> " + result
        if len(words) < max_len:
            #print words,tags
            trees = parseTags(tags,parser)
            for tree in trees:
                print tree
            if debug: print "{} parses for {} {}".format(len(trees),tags,words)
            if len(trees) == 0: fn += 1
            else: 
                tp += 1
                num_parses += len(trees)
            
            for it in range(0,5):
                shuftags = getShuffledTags(tags)
                trees = parseTags(shuftags,parser)
                if debug: print "{} parses for {}".format(len(trees),shuftags)
                if len(trees) > 0: fp += 1
    #print tp,fn,fp
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fmeasure = 2 * recall * precision / (recall + precision)
    quality = {'f-measure': fmeasure, 'recall': recall, 'precision' : precision, 'parses-per-sent': num_parses / tp}
    return quality
        
        
            
