import numpy as np
import pandas as pd
import string
import argparse
import time

import nltk
from nltk.stem import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from wordcloud import WordCloud 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import scipy
from scipy.cluster.hierarchy import ward, average,dendrogram, linkage

import matplotlib.pyplot as plt

from joblib import Parallel, delayed, cpu_count
import seaborn as sns

parser = argparse.ArgumentParser(description='Generate similar courses.')
parser.add_argument('--input', '-input', metavar='input', \
	default='/home/C00219805/Freelance/Upwork/harvard_class_data_fall_2018.csv', \
	help='Input File Location.')
parser.add_argument('--output', '-output', metavar='output', default='/home/C00219805/Freelance/Upwork/', \
	help='Path where output is stored.')
parser.add_argument('--n', '-n', metavar='n', default = 10, \
	help='Required Similar N courses.')
parser.add_argument('--course', '-course', metavar='course', default=None, \
	help='Course Name to which Similar N courses required.')
parser.add_argument('--course1', '-course1', metavar='course1', default=None, \
	help='First Course Name to which Similarity to second Course is required.')
parser.add_argument('--course2', '-course2', metavar='course2', default=None, \
	help='Second Course Name to which Similarity to First Course is required.')
parser.add_argument('--scheme', '-scheme', metavar='scheme', default='tfidf', \
	help='Scheme like tfidf, tf or wordnet sims to incorporate in wrod-sentence matrix building.')
parser.add_argument('--sim_measure', '-sim_measure', metavar='sim_measure', default='cosine', \
	help='Select one among Cosine, Euclidean or Manhattan as distance measure.')
parser.add_argument('--only_nouns', '-only_nouns', metavar='only_nouns', default=True, \
	help='True if you need the course to be represented with only nouns in the description.')

def generate_word_cloud(frequencies):
	wordcloud = WordCloud().generate(frequencies)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()

def word_tokenize(text):
	return tokenizer(text)

def get_trimmed_sentences(sentences):
	sentence_trimmed_dict = {}
	alltokens = []
	nouns = set()
	for key, sent in sentences.iteritems():
		tokens = word_tokenize(sent)
		#print nltk.ne_chunk(nltk.pos_tag(tokens))
		sent_nouns = [word.lower() for word,tag in nltk.pos_tag(tokens) if ((tag == 'NN') | (tag== 'NNP') | (tag== 'NNP') |(tag== 'NNS'))]
		nouns.update(sent_nouns)
		tokens = [x.strip(string.punctuation) for x in sent.split()]
		tokens = [w for w in tokens if not w in stopset]
		alltokens = alltokens + tokens
		sentence_trimmed_dict[key] = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sent_nouns]).strip()

	return sentence_trimmed_dict,alltokens,nouns

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

def get_sentence_vectors(sentence, noun_syns):
	sent_vector = {}
	sent_syns = [tagged_to_synset(token) for token in word_tokenize(sentence) if tagged_to_synset(token)]
	for noun_syn in noun_syns:
		wn_sims = [noun_syn.wup_similarity(ss) for ss in sent_syns]
		if wn_sims:
			 sent_vector[noun_syn.lemmas()[0].name().lower()] = max(wn_sims)
		else:
			 sent_vector[noun_syn.lemmas()[0].name().lower()] = 0
	return sent_vector
def calculate_wordnet_sims(nouns, sentence_trimmed_dict):
	noun_syns = [tagged_to_synset(noun) for noun in nouns if tagged_to_synset(noun)]
	all_sent_vectors = {}
	i =0
	for course, sentence in sentence_trimmed_dict.iteritems():
		i +=1
		print i, ': ', course,': ', sentence
		all_sent_vectors[course] = get_sentence_vectors(sentence, noun_syns)

	matrix = pd.DataFrame(all_sent_vectors)
	matrix = matrix.fillna(0)
	matrix.to_csv('{}/wordnet-matrix.csv'.format(path))
	dtm = matrix.values
	dtm = dtm.transpose()
	return dtm, matrix.columns

def get_word_sentence_matrix(scheme, nouns, sentence_trimmed_dict):
	trim_sents,labels = sentence_trimmed_dict.values(), sentence_trimmed_dict.keys()
	if scheme == 'custom':
		dtm = calculate_wordnet_sims(nouns, sentence_trimmed_dict)
	elif scheme == 'tfidf':
		vectorizer =TfidfVectorizer(min_df = 1, stop_words = 'english')
		dtm = vectorizer.fit_transform(trim_sents)				
		#terms = vectorizer.get_feature_names()
	elif scheme == 'tf':
		vectorizer =CountVectorizer(min_df = 1, stop_words = 'english')
		dtm = vectorizer.fit_transform(trim_sents)				
		#terms = vectorizer.get_feature_names()
	return dtm, labels

def get_similarities(dtm, labels, sim_measure):
	# Try different distance metrics
	if sim_measure == 'cosine':
		dist = 1 - cosine_similarity(dtm)
	elif sim_measure == 'euclidean':
		dist = euclidean_distances(dtm)
	elif sim_measure == 'manhattan':
		dist = manhattan_distances(dtm)
	dist_df = pd.DataFrame((1-dist), columns = labels, index = labels)
	# dist_df = dist_df[sorted(labels)]
	# dist_df = dist_df.sort_index()
	dist_df.to_csv('{}/distance-{}-matrix.csv'.format(path, sim_measure))

	# Hierarchical Clustering		
	get_hierarchical_clustering(dist, labels, path)
	
	return dist_df

def get_course_specific_similarities(course, dtm, labels, sim_measure):
	dtm = dtm.toarray()
	index = labels.index(course)
	# Try different distance metrics
	if sim_measure == 'cosine':
		dist = 1 - cosine_similarity(dtm, [dtm[index]])
	elif sim_measure == 'euclidean':
		dist = euclidean_distances(dtm, [dtm[index]])
	elif sim_measure == 'manhattan':
		dist = manhattan_distances(dtm, [dtm[index]])
	dist_df = pd.DataFrame((1-dist), columns = [course], index = labels)
	# dist_df = dist_df[sorted(labels)]
	# dist_df = dist_df.sort_index()
	dist_df.to_csv('{}/distance-{}-matrix-{}.csv'.format(path, sim_measure, course))

	return dist_df

def get_hierarchical_clustering(dist, labels, path):
	linkage_matrix = ward(dist)   
	print linkage_matrix
	#print linkage_matrix.shape, dist.shape
	fig, ax = plt.subplots(figsize=(15, 20)) # set size
	#print labels
	ax = dendrogram(linkage_matrix, orientation="right", labels=labels) 
	plt.tick_params(\
	    axis= 'x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off')
	plt.savefig('{}/ward_hierarchical_clusters.png'.format(path), dpi=200)
	#plt.show()

def get_sheetname(column):
	if len(column.split(' -')[0]) >31:
		return column.split(' -')[0][0:30].replace(':','')
	else:
		return column.split(' -')[0].replace( ':','')

def get_N_similar_courses(df, n, course_desc_dict, writer , course):	
	similar_courses_dict = {}
	iteratable = [course] if course else df.columns
	course_desc_df = pd.DataFrame(course_desc_dict.items(), columns = ['course', 'long_description'])
	for column in iteratable:	
		df_course = pd.DataFrame({'course': df[column].index, 'similarity': df[column].values})
		if course:
			df_course.merge(course_desc_df, on = 'course', how = 'left').\
				sort_values(by = ['similarity'], ascending = False).\
				to_excel(writer,sheet_name= get_sheetname(column))
		else:
			df_course.merge(course_desc_df, on = 'course', how = 'left').\
				sort_values(by = ['similarity'], ascending = False).head(n).\
				to_excel(writer,sheet_name= get_sheetname(column))
		df_course = df_course.sort_values(by = ['similarity'], ascending = False).head(n)
		df_course['course-sim'] = df_course['course'] + '(' + df_course['similarity'].map(str) + ')'		
		similar_courses_dict[column] = df_course['course-sim'].values
		print column , ': ', df_course['course-sim'].values
	return pd.DataFrame(similar_courses_dict.items())

if __name__ == '__main__':
	start = time.time()
	args = parser.parse_args()
	path = args.output

	# Initializing nltk package related variables
	stopset = set(stopwords.words('english'))
	# adding the word course to stopset
	stopset.add('course')
	tokenizer = TreebankWordTokenizer().tokenize

	train = pd.read_csv(args.input, header = 0)

	# Data Cleaning
	train['long_description'] = train['long_description'].fillna(train['short_description'])
	train['long_description'] = train['long_description'].apply(lambda x: str(x).replace('Description:  ', ''))
	train['long_description'] = train['long_description'].astype(str).str.lower()
	train['course_category'] = train['course_name'].apply(lambda x: x.split(' ')[0])
	train['course_category'] = train['course_category'].astype(str).str.lower()
	train['combined'] = train['long_description'] + ' ' + train['short_description'] + ' ' + train['course_category']
	train['combined'] = train['combined'].astype(str).str.lower()
	print train['combined'].values
	#Check for nulls
	print 'Number of Nulls in data: ',train['course'].isnull().sum()

	sentences = pd.Series(train['combined'].values, index =train['course'].values ).to_dict()
	print 'Number of unique Courses:', len(sentences)

	# WordCloud
	# text = ''
	# for sent in sentences: text += '.'+sent
	# generate_word_cloud(text)

	sentence_trimmed_dict,alltokens,nouns = get_trimmed_sentences(sentences)
	trim_sents,course_labels = sentence_trimmed_dict.values(), sentence_trimmed_dict.keys()

	dtm, labels = get_word_sentence_matrix(args.scheme, nouns, sentence_trimmed_dict)
	
	if args.scheme != 'custom':
		pd.DataFrame(dtm.toarray()).to_csv('{}/word-sentence-matrix-{}.csv'.format(path, args.scheme))	
	
	# HeatMap	
	# sns.heatmap(df)
	excel_name = '{}/coursewise_similarities_{}_{}.xlsx'.format(path, args.scheme, args.sim_measure)
	writer = pd.ExcelWriter(excel_name,engine='xlsxwriter')
			
	#Specific Course
	if args.course:
		dist_df = get_course_specific_similarities(args.course, dtm, labels, args.sim_measure)
	# Top N Similar courses	for all			
	else:
		# Calculate Similarities
		dist_df = get_similarities(dtm, labels, args.sim_measure)
		# Uncomment below if similarity matrix already present
		# df = pd.read_csv('{}/distance-matrix_ordered.csv'.format(path), header =0, index_col =0)
			
	df = get_N_similar_courses(dist_df, args.n, sentences, writer, args.course)
	df.to_csv('{}/similar_courses_top_{}_{}.csv'.format(path,args.n, args.scheme, args.sim_measure))
	print("All Details of the similarity search for the request can be found at: {}".format(excel_name))
	print("Task Completed. Total time taken: {} mins.".format(str((time.time() - start)/ 60)))

	

