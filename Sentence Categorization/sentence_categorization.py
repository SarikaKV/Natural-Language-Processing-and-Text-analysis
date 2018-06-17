import nltk,os,string,operator
from wordcloud import WordCloud 
from os import walk
from os import listdir
from os.path import isfile, join
import time

from joblib import Parallel, delayed, cpu_count

from nltk.stem import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import StanfordNERTagger
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import collections
import numpy as np
import scipy
from scipy.cluster.hierarchy import ward, dendrogram
import enchant as spell
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import pandas as pd
import pickle,spacy

from sentiment_analysis import SentimentAnalysis
from summization import Summarization

input_entity_files = os.path.join(os.path.expanduser('~'),'520','Project','Summarization files','users3')
label1_text = ['food','chicken','pasta']
label2_text = ['service','price','cost','quality']
label3_text = ['ambiance','surrounding','place']
tokenizer = TreebankWordTokenizer().tokenize
stopset = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
wc = WordCloud()
def read_file(file_name):
        """Read the file."""
        with open(file_name, 'r') as file:
            return file.read().decode('utf-8').lower()

def word_tokenize(text):
	return tokenizer(text)

def generate_word_cloud(frequencies):
	wordcloud = WordCloud().generate(frequencies)
	#wordcloud = wc.generate_from_frequencies(dict(sorted(frequencies.items(), key=operator.itemgetter(1))))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()

def get_sentences_with_nouns_adjectives(sentences):
        """Categorizes sentences into only nouns, only adjectives,nouns and adjectives and all others. """ 
        sentences_only_nouns = []
        sentences_only_adjectives = []
        sentences_nouns_adjectives = []
        sentences_others = []
        adjectives = []
        for sent in sentences:
            tagged_words = nltk.pos_tag(nltk.word_tokenize(sent))
            for word,tag in tagged_words : 
                if (tag == 'JJ'):
                    adjectives.append(word.lower())
            nouns = [word.lower() for word,tag in tagged_words   if ((tag == 'NN') | (tag== 'NNS') | (tag== 'NNP') | (tag== 'NNPS'))]
                 
            if any(word.lower() in sent for word in nouns) and any(word.lower() in sent for word in adjectives):
                sentences_nouns_adjectives.append(sent)
            elif any(word.lower() in sent for word in nouns) and not any(word.lower() in sent for word in adjectives):
                sentences_only_nouns.append(sent)
            elif not any(word.lower() in sent for word in nouns) and any(word.lower() in sent for word in adjectives):
                sentences_only_adjectives.append(sent)
            else:
                sentences_others.append(sent)
        return sentences_only_nouns,sentences_only_adjectives,sentences_nouns_adjectives,sentences_others, list(set(adjectives))  

def get_trimmed_sentences(sentences):
	sentence_trimmed_dict = {}
	alltokens = []
	nouns = set()
	for sent in sentences:
		tokens = word_tokenize(sent)
		#print nltk.ne_chunk(nltk.pos_tag(tokens))
		sent_nouns = [word.lower() for word,tag in nltk.pos_tag(tokens) if ((tag == 'NN') | (tag== 'NNP') | (tag== 'NNP') |(tag== 'NNS'))]
		nouns.update(sent_nouns)
		#print nouns
		tokens = [x.strip(string.punctuation) for x in sent.split()]
		tokens = [w for w in tokens if not w in stopset]
		alltokens = alltokens + tokens
		#sentence_trimmed_dict[sent] = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sent_nouns]).strip()
		sentence_trimmed_dict["".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sent_nouns]).strip()] = sent

	return sentence_trimmed_dict,alltokens,nouns
	
def get_bigrams(tokens):
	bigram_finder = list(nltk.bigrams(tokens)) #BigramCollocationFinder.from_words(tokens)
	fdist = nltk.FreqDist(bigram_finder)
	fd ={}
	for k,v in sorted(((c,ng) for ng, c in fdist.items()), reverse = True):
		fd[v[0] +' ' +v[1]] = k
		#print k,v
def tagged_to_synset(word):
	try:
		syns = wn.synsets(word)
		syn = syns[0]
		for s in syns:
				if s.lemmas()[0].name().lower() == word.lower():
					syn = s
					break
		return syn
	except:
		return None
def get_sentence_vectors(sentence, nouns):
	sent_vector = {}
	sent_syns = [tagged_to_synset(token) for token in word_tokenize(sentence) if tagged_to_synset(token)]
	noun_syns = [tagged_to_synset(noun) for noun in nouns if tagged_to_synset(noun)]
	for noun_syn in noun_syns:
		if [noun_syn.wup_similarity(ss) for ss in sent_syns]:
			 sent_vector[noun_syn.lemmas()[0].name().lower()] = max([noun_syn.wup_similarity(ss) for ss in sent_syns])
		else:
			 sent_vector[noun_syn.lemmas()[0].name().lower()] = 0
	return sent_vector
		#print noun_syn,best_score
def get_KMeans_clustering(dtm):
	num_clusters = 5
	km = KMeans(n_clusters=num_clusters)
	km.fit(dtm)
	clusters = km.labels_.tolist()
	print clusters
	sentencesDict = {'original_sent':indexed_sents,'trimmed_sent':trim_sents,'cluster':clusters}		
	frame = pd.DataFrame(sentencesDict, index = [clusters] , columns = ['original_sent', 'trimmed_sent','cluster'])	
	print frame
	print frame['cluster'].value_counts() 
	print("Top terms per cluster:")
	#sort cluster centers by proximity to centroid
	order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

	for i in range(num_clusters):
	    print "Cluster %d words:" % i, ''
	    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
	    	print ind, order_centroids[i, :6], terms[ind]
	    	#print vocab_frame.ix[terms[ind]]
	        #print ' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), ','
def get_hierarchical_clustering(dist):
	linkage_matrix = ward(dist)   
	print linkage_matrix
	print linkage_matrix.shape, dist.shape
	fig, ax = plt.subplots(figsize=(15, 20)) # set size
	ax = dendrogram(linkage_matrix, orientation="right", labels=indexed_sents) 
	plt.tick_params(\
	    axis= 'x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off')

	#plt.tight_layout() #show plot with tight layout
	plt.savefig(input_entity_files +'ward_clusters_2.png', dpi=200)
	plt.show()
def get_label_vector(nouns,label):
	nouns_similarities = {}  
	#label1 = wn.synsets(label1_text)[0]
	label_syns = [tagged_to_synset(l) for l in label if tagged_to_synset(l)]
	for noun in nouns:
			#print 'suggestion for :', noun, d.suggest(noun)
			syns = wn.synsets(noun)
			if syns:
				syn = syns[0]
				for s in syns:
					if s.lemmas()[0].name().lower() == noun.lower():
						syn = s
				#similarity = label1.wup_similarity(syn) if label1.wup_similarity(syn) else 0
				similarity = max([ss.wup_similarity(syn) for ss in label_syns]) if max([ss.wup_similarity(syn) for ss in label_syns]) else 0
				#print syn,similarity, max([ss.wup_similarity(syn) for ss in label_syns])
				if similarity > 0.35:
					nouns_similarities[syn.lemmas()[0].name().lower()] = similarity
				else:
					nouns_similarities[syn.lemmas()[0].name().lower()] = similarity
			else:
				nouns_similarities[noun] = 0
	return collections.OrderedDict(sorted(nouns_similarities.items())).values()
def get_sentence_label_weights(sentence):
	sent_syns = [tagged_to_synset(token) for token in word_tokenize(sentence) if tagged_to_synset(token)]
def processFiles(file,dirpath):
			file_name = os.path.splitext(file)[0]
			print file_name
			text = read_file(dirpath+'/'+file)
			
			#generate_word_cloud(text)
			print '#sentences:',len(sent_tokenize(text))
			#If you want to work on full sentences
			#sentences.append(len(sent_tokenize(text)))
			#print sum(sentences)/len(sentences)
			sentences = [chunk.text.lower() for chunk in nlp(text).noun_chunks  if nlp.vocab[chunk.lemma_].prob < -8]
			sentences_only_nouns,sentences_only_adjectives,sentences_nouns_adjectives,sentences_others,adjectives =  get_sentences_with_nouns_adjectives(sentences)
			print len(sentences_only_nouns),len(sentences_only_adjectives),len(sentences_nouns_adjectives),len(sentences_others)
			interested_sentences = sentences_nouns_adjectives
			
			output = get_trimmed_sentences(interested_sentences)[0]
			original_sents,trim_sents = get_trimmed_sentences(interested_sentences)[0].values(),get_trimmed_sentences(interested_sentences)[0].keys()

			#If you want to test with TFID vectorization
			#vectorizer =TfidfVectorizer(min_df = 1, stop_words = 'english')
			#dtm = vectorizer.fit_transform(trim_sents)				
			#vocab_frame = pd.DataFrame({'words': get_trimmed_sentences(interested_sentences)[1]})
			#terms = vectorizer.get_feature_names()
			
			nouns = get_trimmed_sentences(interested_sentences)[2]
			print len(nouns)
			all_sent_vectors = {}
			for sentence in trim_sents:
				all_sent_vectors[sentence] = get_sentence_vectors(sentence, nouns)
			matrix = pd.DataFrame(all_sent_vectors)
			matrix = matrix.fillna(0)
			#Uncomment below for generating new pickle files
			#matrix.to_pickle(os.path.join(os.path.expanduser('~'),'520','Project','Summarization files','output') + '/matrix_only_noun_sents_3.pkl')
			#matrix = pd.read_pickle(input_entity_files + 'matrix.pkl')
			indexed_sents = matrix.columns
			dtm = scipy.sparse.csr_matrix(matrix.values)
			dtm = dtm.transpose()

			#KMeans Clustering				
			#get_KMeans_clustering(dtm)

			#Hierarchical Clustering
			#dist = 1 - cosine_similarity(dtm)			
			#get_hierarchical_clustering(dist)

			label = label2_text

			origs = [output[s] for s in matrix.columns]
			#Calculate the dot product
			df = pd.DataFrame({'original_sents': origs,'sentences':matrix.columns,'weights':np.array(dtm.todense().dot(np.transpose(np.array(get_label_vector(matrix.index,label)))))[0].tolist()})
			df = df.sort_values(by=['weights'],ascending = False)
			df = df[df['weights'] >0]
			df.to_csv(os.path.join(os.path.expanduser('~'),'520','Project','Summarization files','output')  +'/'+label[0]+'_sents_'+file_name+'.csv', sep=',',encoding='utf-8')
			print df
			#Summarizion
			text = ''
			for sent in df['original_sents'].values: text += '.'+sent
				
			summarize = Summarization(text,None,senti) 
			final_summary1,final_summary2,final_summary3,neg_final_summary1,neg_final_summary2,neg_final_summary3,counts,eigen_explo= summarize.get_summaries()
			print final_summary1,final_summary2,final_summary3,neg_final_summary1,neg_final_summary2,neg_final_summary3,counts,eigen_explo
if __name__ == "__main__":
	d = spell.request_dict("en_US")
	nlp = spacy.load('en')   

	senti = SentimentAnalysis()  

	start_time = time.time()
	filenames = [ f for f in listdir(input_entity_files) if isfile(join(input_entity_files, f))]
	#Parellel on CPU cores
	#Parallel(n_jobs=cpu_count() - 1, verbose=10, backend="multiprocessing", batch_size="auto")(delayed(processFiles)(fileName,input_entity_files) for fileName in filenames)
	#for (dirpath, dirnames, filenames) in walk(input_entity_files):
	for file in filenames:
		processFiles(file,input_entity_files)

	print("Time taken --- %s seconds ---" % (time.time() - start_time))
	
	
	

	
   
    