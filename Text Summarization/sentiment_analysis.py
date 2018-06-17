import nltk,os
from nltk.stem import *


bad_words_file = os.path.join(os.path.expanduser('~'),'Summarization files','Dictionaries','negative-words.txt')
SEED_LIST = {"good": 1, "excellent": 1,"cool":1, "nice": 1,"fabulous":1,"fantastic":1,"supreme":1,"durable":1,
                 "important":1,"reliable": 1,"amazing":1, "great":  1,"beautiful":1,"comfortable":1,"phenomenal":1,
                 "perfect":1,"happy":1,"superb":1,"pretty":1,"portable" :1,"fine":1,"recommended":1,"unbelievable":1,"quick":1,
                 "stylish":1,"clear":1,"unique":1,"cheap":1,"exceptional":1,"impressive":1,"impressed":1,"suitable":1,"nearby":1,
                 "rich":1,"reasonable":1,"remarkable": 1,"smooth":1,"soft":1,"worth":1,"tasty":1,"friendly":1,"lovely":1,
                 "downside":-1,"overpriced":-1, "awful":-1,"bland":-1,
                 "uncomfortable": -1,"frustrating":-1,"crappy":-1, "not" : -1,"cant":-1,"bad":-1,"unreliable":-1,"poor":-1,"ugly":-1,
                 "worst":-1,"wrong":-1,"distorted":-1,"expensive":-1,"exaggerated":-1,"low":-1, "sad":-1,"hard":-1,"disappointed":-1}

class SentimentAnalysis():
    
    def __init__(self):
        self.build_seed_list()

    def build_seed_list(self):
        """Build word- sentiment list from the initial seed list."""
        #First assign negative sentiment to all the words in external file
        text_file = open(bad_words_file, "r")
        lines = text_file.read().split('\n')
        for word in lines : SEED_LIST[word] = -1
        
        for word in SEED_LIST.keys():
            syns = wordnet.wordnet.synsets(word)
            for syn in syns:
                for l in syn.lemmas():          
                    if l.name() not in SEED_LIST:
                       sentiment = SEED_LIST[word]
                       SEED_LIST[l.name()] = sentiment    
                    if l.antonyms():
                           ant =l.antonyms()[0].name() 
                           if ant  not in SEED_LIST:
                               if SEED_LIST[word] == 1:  sentiment =-1
                               else: sentiment = 1   
                               SEED_LIST[ant] = sentiment   
                               
    def print_seed_list(self):
        print SEED_LIST
    
    def get_word_orientation(self,opinion_word):
        """Return orientation or sentiment of given word."""
        sentiment = 0
        if opinion_word in SEED_LIST:  sentiment = SEED_LIST[opinion_word]    
        else:
            syns = wordnet.wordnet.synsets(opinion_word)
            for syn in syns:
                for l in syn.lemmas():           
                    if l.name() in SEED_LIST:
                       sentiment = SEED_LIST[l.name()]
                       if l.antonyms():
                           ant =l.antonyms()[0].name() 
                           if ant  in SEED_LIST:
                               if SEED_LIST[ant] == 1:  sentiment =-1
                               else: sentiment = 1            
                       SEED_LIST[opinion_word] = sentiment         
        return sentiment       
    
    def get_sentence_orientation(self,sentences):
        """Return orientation or sentiment of given sentence."""
        positive_sentences = []
        negative_sentences = []
        neutral_sentences = []
        text_file = open(bad_words_file, "r")
        lines = text_file.read().split('\n')
        
        for sent in sentences:
                for word in sent.split(' '): 
                    if word.encode('utf-8')   and  word.encode('utf-8') in lines:
                        negative_sentences.append(sent)
          
                orientation = 0
                #all words of the chunks will be considered as opinion words not just adjectives
                opinion_words = [word.lower() for word,tag in nltk.pos_tag(nltk.word_tokenize(sent)) if ((tag == 'JJ') | (tag== 'JJR') | (tag== 'JJS'))]
                for opinion in opinion_words:   orientation += self.get_word_orientation(opinion)
                         
                if orientation >= 1: positive_sentences.append(sent)      
                #elif orientation <  0:  negative_sentences.append(sent)
                elif sent not in negative_sentences: neutral_sentences.append(sent)              
        return positive_sentences,negative_sentences,neutral_sentences
    
