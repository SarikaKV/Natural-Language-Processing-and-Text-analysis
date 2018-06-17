
import os, time
from os import walk
import sys, codecs, re, json, csv
from __builtin__ import file

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.utils.extmath import randomized_svd

import nltk, spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from sentiment_analysis import SentimentAnalysis
import spell_enchant as spell

import _sysconfigdata_nd
from yaml import representer

NUM_TOPICS = 10
NUM_ENTITIES = 5
NUM_SENTENCES_PER_SUMMARY = 10
NUM_FOOD_SENTENCES_PER_SUMMARY = int(0.6*NUM_SENTENCES_PER_SUMMARY)
INDEX =  range(0,180)
input_files = os.path.join(os.path.expanduser('~'),'Summarization files','LDA')
input_entity_files = os.path.join(os.path.expanduser('~'),'Summarization files','users')

def read_file(file_name):
        """Read the file."""
        with open(file_name, 'r') as file:
            return file.read().decode('utf-8')
        
def translate_non_alphanumerics(to_translate, translate_to=u''):
        """Translates letters and digits to unicode characters."""
        not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
        translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
        return to_translate.translate(translate_table)

def get_top_topics_topicwords(df):
        topic_words = []
        for words in df['topicWords'].values:
            for word in words.split(' '):
                topic_words.append(str.replace(word,'_',' '))
        return df['topicName'].values, topic_words
    
class Summarization():
    
    sentences_c = []
   
    def __init__(self,text,topic_words,senti):
        self.text = text
        self.topic_words  = topic_words  
        self.senti = senti   
        self.s = ['the', 'an','a']
   
    def get_sentences_with_specific_words(self,sentences):
        """Return sentence if it has specific words.."""
        words = self.topic_words
        result = [sent for sent in sentences  if any(word.lower() in sent for word in words)]
        return result  
    
    def get_sentences_with_nouns_adjectives(self,sentences):
        """Categorizes sentences into only nouns, only adjectives,nouns and adjectives and all others. """ 
        sentences_only_nouns = []
        sentences_only_adjectives = []
        sentences_nouns_adjectives = []
        sentences_others = []
        adjectives = []
        
        for sent in sentences:
            sent = sent.split(' ')
            text = ''
            for s in sent:  
                text += s.lower() + ' '
            tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
            for word,tag in tagged_words : 
                if (tag == 'JJ'):
                    adjectives.append(word.lower())
            nouns = [word.lower() for word,tag in tagged_words   if ((tag == 'NN') | (tag== 'NG') | (tag== 'NNP') | (tag== 'NNPS'))]
                 
            if any(word.lower() in sent for word in nouns) and any(word.lower() in sent for word in adjectives):
                sentences_nouns_adjectives.append(text)
            elif any(word.lower() in sent for word in nouns) and not any(word.lower() in sent for word in adjectives):
                sentences_only_nouns.append(text)
            elif not any(word.lower() in sent for word in nouns) and any(word.lower() in sent for word in adjectives):
                sentences_only_adjectives.append(text)
            else:
                sentences_others.append(text)
        return sentences_only_nouns,sentences_only_adjectives,sentences_nouns_adjectives,sentences_others, list(set(adjectives))   
    
    def get_required_sentences(self):
        global sentences_c 
        #sentences = re.split('[?.,]', text)
        sentences = [chunk.text.lower() for chunk in nlp(self.text).noun_chunks  if nlp.vocab[chunk.lemma_].prob < -8]
        nounchunkscount = len(sentences)
        sentences_c = sentences
        #Remove special characters
        sentences = [translate_non_alphanumerics(sent) for sent in sentences] 
        #Retain only those sentences with LDA topic words
        sentences = self.get_sentences_with_specific_words(sentences)
        ##Pre-process sentences before summarization
        sentences_only_nouns,sentences_only_adjectives,sentences_nouns_adjectives,sentences_others,adjectives =  self.get_sentences_with_nouns_adjectives(sentences)
        #Retain only sentences with positive sentiment and neutral sentiment 
        positive_sentences,negative_sentences,neutral_sentences= self.senti.get_sentence_orientation(sentences_nouns_adjectives  + sentences_only_adjectives+ sentences_only_nouns)
        print positive_sentences
        print negative_sentences
        print neutral_sentences
        sentences = positive_sentences + neutral_sentences
        f,o = self.get_category_sentences(sentences)
        counts = [nounchunkscount,len(sentences), len(positive_sentences),len(negative_sentences),len(neutral_sentences),len(f),len(o)]
        return f,o,counts
            
    def run_lsa_summarizer(self,sentences,num_sents_per_summary):
        """Performs regular LSA using TruncatedSVD."""
        final_summary = ''
        result_len = 0
        if len(sentences) < num_sents_per_summary:
            result_len = len(sentences)
            for sent in sentences:
                final_summary  += sent.lower().strip() +'.' 
        else:
            vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
            dtm = vectorizer.fit_transform(sentences)
            indexed_sents = sentences
            if len(vectorizer.get_feature_names()) <= num_sents_per_summary:
                indexed_sents = sentences_c
                dtm = vectorizer.fit_transform(sentences_c)
            #Use original sentences in index not stemmed output         
            #Fit LSA. Use algorithm = randomized for large  datasets.
            lsa = TruncatedSVD(num_sents_per_summary, algorithm = 'randomized')
            dtm_lsa = lsa.fit_transform(dtm)
            dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
            index_names = ["component_" + str(i) for  i  in range( num_sents_per_summary)] 
            
            ##############Add this in output statistics
            explained_variance = lsa.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
            ###################
            #Sentence Matrix       #Use original sentences in index not stemmed output         
            df_sent = pd.DataFrame(dtm_lsa, index = indexed_sents, columns = index_names)
            
            summary = [df_sent.sort_values(by =index_name,ascending=False).reset_index()['index'].iloc[0].strip() + '.' for index_name in index_names]
            summary = set(summary)
            for sent in set(summary):
                final_summary +=' '.join(filter(lambda w: not w in self.s,sent.split())).lower().strip() 
            result_len = len(summary)
        return final_summary,result_len 
        
    def perform_regular_svd(self,sentences,num_sents_per_summary):
        """Performs regular SVD  using Matrix decomposition."""
        final_summary = ''
        result_len = 0
        eigen_explo = []
        if len(sentences) < num_sents_per_summary:
            #print 'no eigens'
            result_len = len(sentences)
            for sent in sentences:
                final_summary  += sent.lower().strip() +'.'  
        else:
            #Performing general SVD without using LSA package
            vectorizer =CountVectorizer(min_df = 1, stop_words = 'english')
            dtm = vectorizer.fit_transform(sentences)
            indexed_sents = sentences
            if len(vectorizer.get_feature_names()) <= num_sents_per_summary:
                indexed_sents = sentences_c
                dtm = vectorizer.fit_transform(sentences_c)
            summary = []
            dtmT = dtm.transpose()
            U, Sigma, VT = randomized_svd(dtmT, n_components=num_sents_per_summary,
                                              n_iter=5,
                                              random_state=None)
            index_names = ["component_" + str(i) for  i  in range( num_sents_per_summary)]
            V = VT.transpose()
            Sigma_matrix =  np.diag(Sigma)
            eigen_explo = self.check_eigen_value_explosion(Sigma)
            df_sent_tran = pd.DataFrame(VT, index =index_names, columns =indexed_sents )
            df_sent = pd.DataFrame(V, index = indexed_sents, columns = index_names)
            #Identifying the sentences based on top three concepts
            summary = [df_sent.sort_values(by =index_name,ascending=False).reset_index()['index'].iloc[0].strip() + '.' for index_name in index_names] 
            summary = set(summary)
            result_len = len(summary)
            for sent in summary:
                final_summary +=  ' '.join(filter(lambda w: not w in self.s,sent.split())).lower().strip() 
        #print eigen_explo
        return final_summary, result_len,eigen_explo   
    
    def modified_perform_regular_svd(self,sentences,num_sents_per_summary):
        """LSA output modified with sum of product of 
        squares of eigen valesa and corresponding sentence weights.."""
        final_summary = ''
        result_len = 0
        if len(sentences) < num_sents_per_summary:
            result_len = len(sentences)
            for sent in sentences:
                final_summary  += sent.lower().strip() +'.'     
        else:
            #Performing general SVD without using LSA package
            vectorizer =CountVectorizer(min_df = 1, stop_words = 'english')
            dtm = vectorizer.fit_transform(sentences)
            indexed_sents = sentences
            if len(vectorizer.get_feature_names()) <= 3:
                indexed_sents = sentences_c
                dtm = vectorizer.fit_transform(sentences_c)
            summary = []
            dtmT = dtm.transpose()
            U, Sigma, VT = randomized_svd(dtmT, n_components=100,
                                              n_iter=5,
                                              random_state=None)
            V = VT.transpose()
            sent_mod =  np.dot(np.square(V),np.square(Sigma))
            df_sent_mod = pd.DataFrame( np.matrix(sent_mod).transpose(), index = indexed_sents, columns =["index"] )
            df_sent_mod = df_sent_mod.sort_values(by = 'index',ascending= False).reset_index()
            df_sent_mod = df_sent_mod[:num_sents_per_summary]
            summary =  df_sent_mod['level_0'].tolist()
            #summary = set(summary)
            result_len = num_sents_per_summary
            #Identifying top sentences  
            for sent in summary:
                    final_summary += ' '.join(filter(lambda w: not w in self.s,sent.split())).lower().strip() + '.'         
       
        return final_summary,result_len
    def check_eigen_value_explosion (self,sigma):
        sigma2 = np.append(sigma[1:],1)
        eigen_explo= 100*(sigma-sigma2)/sigma
        return  np.round(eigen_explo[:-1],0)
    def category_words(self):
        """Return higherTopics from external file."""
        #df_categories= pd.DataFrame.from_csv(os.path.join(os.path.expanduser('~'),'Summarization files','Dictionaries','categories.csv'))
        #return dict((row[0],row[1].split(' ')) for index, row in df_categories.reset_index().iterrows())
        df_categories= pd.DataFrame.from_csv(os.path.join(os.path.expanduser('~'),'Summarization files','LDA','kmeans_categories_categorywords.csv'))
        return dict((row[0],str.replace(row[1],'_',' ').split(',')) for index, row in df_categories.reset_index().iterrows())
    
    def get_category_sentences(self,sentences):
        """Categorize sentences to higher Topics."""
        categories = self.category_words()
        cat0_sentences = []
        cat1_sentences = []
        other_sentences_final = []
        for sent in sentences:
              sent = sent.split(' ')
              text = ''
              for s in sent: text += s + ' '
              for word in sent :
                  if word.lower() in categories[0] and word is not ' ':
                      cat0_sentences.append(text)
                      break
                  else:
                      if text not in cat0_sentences and text not in cat1_sentences:  cat1_sentences.append(text)               
        other_sentences_final = [sent for sent in cat1_sentences  if  sent not in cat0_sentences]
        return cat0_sentences,other_sentences_final
    
#     def get_category_sentences(self,sentences):
#         """Categorize sentences to higher Topics."""
#         categories = self.category_words()
#         food_sentences = []
#         other_sentences = []
#         other_sentences_final = []
#         for sent in sentences:
#               sent = sent.split(' ')
#               text = ''
#               for s in sent: text += s + ' '
#               for word in sent :
#                   if word.lower() in categories['food '] and word is not ' ':
#                       food_sentences.append(text)
#                       break
#                   else:
#                       if text not in food_sentences and text not in other_sentences:  other_sentences.append(text)               
#         other_sentences_final = [sent for sent in other_sentences  if  sent not in food_sentences]
#         return food_sentences,other_sentences_final
    
    def get_summaries(self):
        """Summarizes given sentences."""
        food_sentences,other_sentences,counts = self.get_required_sentences()
        food_summary1,food_sents1 = self.run_lsa_summarizer(food_sentences,NUM_FOOD_SENTENCES_PER_SUMMARY)
        final_summary1 = food_summary1  + ' ' +  self.run_lsa_summarizer(other_sentences,NUM_SENTENCES_PER_SUMMARY - food_sents1)[0]
        print spell.spellcheck(final_summary1)
        
        food_summary2,food_sents2,eigen_explo = self.perform_regular_svd(food_sentences,NUM_FOOD_SENTENCES_PER_SUMMARY)
        final_summary2 =food_summary2+ ' ' + self.perform_regular_svd(other_sentences,NUM_SENTENCES_PER_SUMMARY - food_sents2)[0]
        print spell.spellcheck(final_summary2)
        
        food_summary3,food_sents3 = self.modified_perform_regular_svd(food_sentences,NUM_FOOD_SENTENCES_PER_SUMMARY)
        final_summary3 = food_summary3+ ' ' + self.modified_perform_regular_svd(other_sentences,NUM_SENTENCES_PER_SUMMARY - food_sents3)[0]
        print spell.spellcheck(final_summary3)
        #print eigen_explo
        return final_summary1,final_summary2, final_summary3,counts,eigen_explo

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    start_time = time.time()
    nlp = spacy.load('en')    
    senti = SentimentAnalysis()  
    df_entities = pd.DataFrame.from_csv('/home/sarika/Summarization files/Dictionaries/entity_id_names_krowd_40percentile.csv')  
    df = pd.read_json(input_files+'/input_entity_json.json')
    df = df[df['id'].isin(df_entities['id'].values)]
    df = pd.merge(df,df_entities, left_on = 'id', right_on = 'id', how = 'inner')  
    df_topics_words= pd.DataFrame.from_csv(input_files+'/topics_words.csv')
    df_topics_words = df_topics_words.dropna().reset_index()
    df_topics_words['topiId'] = df_topics_words['topiId'].astype(int)
    i =0
    print 'Started at ',str(time.ctime(int(time.time())))
    with open('/home/sarika/Summarization files/summary_500_kmeans_run2_04_06_17.csv','w') as outfile:
        writer = csv.writer(outfile, delimiter = ',', lineterminator = '\n',)
        writer.writerow(['docID','docName','entityName','LDA Topics','Summary (Regular SVD)' ,'Summary (Truncated SVD)'
                         ,'Summary (Eigen value weighted sentence identification SVD)','#reviews','#sentences','#nounchunks'
                         ,'#nounchunks considered','#positive sents','#negative sents','#neutral sents','#food sents','#non-food sents'
                         ,'Eigen Explosion'])
        for (dirpath, dirnames, filenames) in walk(input_entity_files):
            for file in filenames:
                if i < NUM_ENTITIES and str.replace( str.replace(file,'entity_', ' '),'.txt', ' ').strip() in df['id'].values:
                    print i,str.replace( str.replace(file,'entity_', ' '),'.txt', ' ')             
                    #Get topic distribution for this entity
                    df_entity_row = df.loc[df['id'] == str.replace( str.replace(file,'entity_', ' '),'.txt', ' ').strip()]
                    print df_entity_row.reset_index()['display_name'][0]
                    topics_array = np.asarray(df_entity_row['topicDistributions'])
                    df_topics = pd.DataFrame(topics_array[0],index = INDEX,columns = ['topicWeight'])
                    df_topics =  df_topics.reset_index()
                    df_topics['topicWeight'] =  df_topics['topicWeight'].astype(float)
                    df_topics = df_topics.sort_values(['topicWeight'], ascending = [False])
                    df_merged = pd.merge(df_topics,df_topics_words, left_on = 'index', right_on = 'topiId', how = 'inner')    
                    #Get all topics and topic words
                    df_top_topics = df_merged.head(NUM_TOPICS)
                    topics_list, topic_words = get_top_topics_topicwords( df_top_topics)
                
                    #Rank the sentences and summarize
                    text = read_file(dirpath+'/'+file)
                    summarize = Summarization( text, topic_words,senti)               
                    final_summary1,final_summary2,final_summary3,counts,eigen_explo= summarize.get_summaries()
                    writer.writerow([i,str.replace( str.replace(file,'entity_', ' '),'.txt', ' '),df_entity_row.reset_index()['display_name'][0],topics_list,final_summary1,final_summary2,final_summary3
                                     , len(text.splitlines()),len([x for x in map(str.strip, text.encode('utf-8').split('.')) if x])
                                     ,counts[0],counts[1],counts[2],counts[3],counts[4],counts[5],counts[6],eigen_explo])
                    i = i +1
    print("--- %s minutes ---" % str((time.time() - start_time)/60))               
                   