import codecs
import random
import time
from nltk.tokenize import word_tokenize
import copy 
import random
import gc


#===================================================================================================================================================#

# This is HMM Part Of Speech tagging class, Tasks done by this class:-
# 1 - //Training\\ - This method extracts all the frequencies and counts and stores them in the appropriate fields.
#     Exact what is being counted has been mentioned in the code below. Once all the data has been extracted from the training set 
#	  this method call make_emmision_matrix(), make_transition_matrix() make_transition_matrix_2(). Completing the
#	  whole training process.
#	  /////Implemented in - train()\\\\\  

# 2 - //Making emmision matrix\\ - This is the probability matrix which keeps track of all the probabilities for each word 
#	  being associated with any unique tag. 
#     Smoothing used: Add-1 or Laplace's smoothing
#     /////Implemented in - make_emmision_matrix()\\\\\

# 3 - // Making 2nd order Transition Matrix\\ - This is the transition matrix which keeps track of all the probabilities of  
#     all possible pair of transitions  i.e. probability of tag2 coming after tag1 in a pair of (tag1, tag2).
#     Smoothing used: Add-1 or Laplace's smoothing
#     /////Implemented in - make_transition_matrix()\\\\\

# 4 - // Making 3rd order Transition Matrix\\ - This is the transition matrix which keeps track of all the probabilities of  
#     all possible triplets of transitions  i.e. probability of tag3 coming after tag1 and tag2 in a triplet of (tag1, tag2, tag3).
#     Smoothing used: Add-1 or Laplace's smoothing
#     /////Implemented in - make_transition_matrix_2()\\\\\

# 5 - // 2nd order Viterbi algorithm\\ - This is the algorithm used to decode the best order of part of speech tags from the given
#     sequence of words. This implementation takes the previous state as context to predict the probability of the current state.
#     Smoothing used: Add-1 or Laplace's smoothing
#     /////Implemented in - viterbi_bigram()\\\\\

# 6 - // 3rd order Viterbi algorithm\\ - This is the algorithm used to decode the best order of part of speech tags from the given
#     sequence of words. This implementation takes the previous two states as context to predict the probability of the current state.
#     Smoothing used: Add-1 or Laplace's smoothing
#     /////Implemented in - viterbi_bigram_2()\\\\\

class HMM_POS_Tagging:

	## Initializing the necessory fields
	num_sentences = 0                          # Keeping track of number of sentences
	total_num_words = 0                        # Keeping track of total number of words
	total_num_tags = 0	                       # Keeping track of total number of tags
	word_tag_count = {}                        # Keeping track of frequencies of word associated to a tag
	tag_count = {}                             # Keeping track of frequency of appearence of each tag
	unique_tags = set([])                      # Storing unique tags
	unique_words = set([])                     # Storing unique words
	tag_tag_count = {}                         # Keeping track of frequency of appearance each tag_tag pair
	tag_tag_tag_count = {}                     # Keeping track of frequency of appearance each tag_tag_tag triplet
	max_sentence_length = 0                    # Maximum sentence length
	min_sentence_length = 9999999999           # Minimum sentence length
	emission_matrix = {}                       # Emmision matrix stored as a dictionary
	transition_matrix = {}                     # 2nd order transition matrix, stored in a dictionary
	transition_matrix_2 = {}                   # 3rd order transition matrix, stored in a dictionary
	viterbi_cache = []                         # 2nd order Viterbi 2D matrix
	viterbi_cache_2 = []                       # 3rd order Viterbi 2D matrix


	## Method used to train the model with the given set of training sentences
	## sentences - list of strings
	def train(self, sentences):
		

		### Traversing over all the sentences
		self.num_sentences = len(sentences)
		temp_num_sents = len(sentences)
		for i in range(temp_num_sents):

			#### Splitting sentnces at " "(spaces) for getting sequence of words
			sents = sentences[i].split(" ")
			num_words = len(sents)


			#### Discarding sentences with less than 2 words/space separated seqs
			if(num_words<=2):
				self.num_sentences-=1
			else:

				##### Updating max and min length of sentence
				self.max_sentence_length = max(self.max_sentence_length,num_words)
				self.min_sentence_length = min(self.min_sentence_length,num_words)


				##### Traversing over words of the sentence
				tags = []
				prev_tag = ""
				prev_prev_tag = ""
				for j in range(num_words):

					###### Moving on if the word tag is missing
					if(sents[j].find("_")==-1):
						continue


					###### Separating word from it's tag 
					word_tag = sents[j].split("_")
					word = word_tag[0]
					tag = word_tag[1]
					word_tag = (word,tag)


					###### Adding to total words and tags
					self.total_num_words+=1
					self.total_num_tags+=1

					###### Appending to tags of this sentence
					tags.append(tag)


					###### Incrementing (word, tag) count
					if(self.word_tag_count.get(word_tag) == None):
						self.word_tag_count[word_tag] = 1
					else:
						self.word_tag_count[word_tag] += 1


					###### Incrementing tag count
					if(self.tag_count.get(tag) == None):
						self.tag_count[tag] = 1
					else:
						self.tag_count[tag] += 1

					###### Appending to unique_words/tags set to get unique words and tags
					self.unique_words.add(word)
					self.unique_tags.add(tag)


					###### Incrementing (tag, tag) count by analysing tags sequences of the sentence
					if(prev_tag == ""):
						pass
					else:
						tag_tag = (prev_tag,tag)
						if(self.tag_tag_count.get(tag_tag) == None):
							self.tag_tag_count[tag_tag] = 1
						else:
							self.tag_tag_count[tag_tag] +=1

						prev_tag = tag


					###### Incrementing (tag, tag, tag) count by analysing tags sequences of the sentence
					if(prev_prev_tag == ""):
						pass
					else:
						tag_tag_tag = (prev_prev_tag, prev_tag, tag)
						if(self.tag_tag_tag_count.get(tag_tag_tag) == None):
							self.tag_tag_tag_count[tag_tag_tag] = 1
						else:
							self.tag_tag_tag_count[tag_tag_tag] += 1


					prev_prev_tag = prev_tag
					prev_tag = tag



		### Making emmision and transition matrices
		self.make_emmision_matrix()
		self.make_transition_matrix()
		# self.make_transition_matrix_2()


	## Method to create the emmision matrix
	def make_emmision_matrix(self):
		for word in self.unique_words:
			for tag in self.unique_tags:

				##### Applying Add-1 or Laplace's smoothing
				if(self.word_tag_count.get((word,tag)) != None):
					self.emission_matrix[(word, tag)] = (self.word_tag_count[(word,tag)] + 1)/(self.tag_count[tag] + len(self.unique_words))
				else:
					self.emission_matrix[(word, tag)] = 1/(self.tag_count[tag] + len(self.unique_words))


	## Making 2nd order transition matrix
	def make_transition_matrix(self):
		for tag1 in self.unique_tags:
			for tag2 in self.unique_tags:

				##### Applying Add-1 or Laplace's smoothing
				if(self.tag_tag_count.get((tag1,tag2)) != None):
					self.transition_matrix[(tag1, tag2)] = (self.tag_tag_count[(tag1,tag2)]+1)/(self.tag_count[tag1] + len(self.unique_tags))
				else:
					self.transition_matrix[(tag1, tag2)] = 1/(self.tag_count[tag1] + len(self.unique_tags))


	## Making 3rd order transition matrix
	def make_transition_matrix_2(self):
		for tag1 in self.unique_tags:
			for tag2 in self.unique_tags:
				for tag3 in self.unique_tags:

					##### Applying Add-1 or Laplace's smoothing
					num_tags = len(self.unique_tags)
					if(self.tag_tag_count.get((tag1,tag2)) != None):
						if(self.tag_tag_tag_count.get((tag1,tag2,tag3)) != None ):
							self.transition_matrix_2[(tag1, tag2, tag3)] = (self.tag_tag_tag_count[(tag1,tag2,tag3)]+1)/(self.tag_tag_count[(tag1,tag2)] + num_tags*num_tags)
						else:
							self.transition_matrix_2[(tag1, tag2, tag3)] = 1/(self.tag_tag_count[(tag1,tag2)] + num_tags*num_tags)
					else:
						self.transition_matrix_2[(tag1, tag2, tag3)] = 1/(num_tags*num_tags)

	
	## Applying 2nd order viterbi algorithm
	## sentence = string
	def viterbi_bigram(self, sentence):

		### Using NLTK's word tokenizer to tokenize our sentence into words
		sentence = word_tokenize(sentence)


		### Inatializing 2D matrix of dimension (Sentence_Length X Num_Tags)
		self.viterbi_cache = []
		prob_guess = {}
		for tag in self.unique_tags:
			prob_guess[tag] = [0,None]
		for word in sentence:
			self.viterbi_cache.append([word,copy.deepcopy(prob_guess)])


		### Applying viterbi alogorithm ###
		

		### Traversing over all the words of the sentence
		l = len(self.viterbi_cache)
		for i in range(l):


			#### Traversing over all the tags this word can have
			for token in self.viterbi_cache[i][1]:
				word = self.viterbi_cache[i][0]


				##### For first word just take the emission probability 
				if i == 0:
					if(self.emission_matrix.get((word,token)) != None):
						self.viterbi_cache[i][1][token][0] = self.emission_matrix[(word,token)]
					else:
						self.viterbi_cache[i][1][token][0] = 1/(self.tag_count[token] + len(self.unique_words))
				else:
					maxx = None
					guess = None
					c = None


					###### Traversing over all the prev tags and taking the maximum probability path
					for k in self.viterbi_cache[i-1][1]:

						####### Calculating probability of current state given previous state
						c = self.viterbi_cache[i-1][1][k][0]*self.transition_matrix[(k,token)]
						if maxx == None or c > maxx:
							maxx = c
							guess = k
					if(self.emission_matrix.get((word,token)) != None):
						maxx *= self.emission_matrix[(word,token)]
					else:
						maxx *= 1/(self.tag_count[token] + len(self.unique_words))


					###### Updating the max probability and the previous state which gave the maximum probability
					self.viterbi_cache[i][1][token][0] = maxx
					self.viterbi_cache[i][1][token][1] = guess


		### Finding the final most probable sequence of tags for our sequence of words
		tokens = []
		token = None

		### Finding the sequence with maximum probability and then backtracking to get the most probable sequence of tags
		for i in range(l-1,-1,-1):
			if token == None:
				maxx = None
				guess = None
				for k in self.viterbi_cache[i][1]:
					if maxx == None or self.viterbi_cache[i][1][k][0] > maxx:
						maxx = self.viterbi_cache[i][1][k][0]
						token = self.viterbi_cache[i][1][k][1]
						guess = k
				tokens.append(guess)
			else:
				tokens.append(token)
				token = self.viterbi_cache[i][1][token][1]
		tokens.reverse()
		gc.collect()
		return tokens


	## Applying 3rd order viterbi algorithm
	## sentence = string
	def viterbi_trigram(self, sentence):


		### Using NLTK's word tokenizer to tokenize our sentence into words
		sentence = word_tokenize(sentence)


		### Inatializing 2D matrix of dimension (Sentence_Length X Num_Tags)
		self.viterbi_cache_2 = []
		prob_guess = {}
		for tag in self.unique_tags:
			prob_guess[tag] = [0,None]
		for word in sentence:
			self.viterbi_cache_2.append([word,copy.deepcopy(prob_guess)])


		### Applying viterbi ###


		### Traversing over all the words of the sentence
		l = len(self.viterbi_cache_2)
		for i in range(l):


			#### Traversing over all the tags this word can have
			for token in self.viterbi_cache_2[i][1]:
				word = self.viterbi_cache_2[i][0]

				if i == 0:

					###### Giving emmision probability to the first words of the sentence 
					self.viterbi_cache_2[i][1][token][0] = self.emission_matrix[(word,token)]
				elif(i == 1):


					###### Using context as previous word for the 2nd word, similar to 2nd order viterbi 
					maxx = None
					guess = None
					c = None


					###### Traversing on all the previous tags which are treating as context 
					for k in self.viterbi_cache_2[i-1][1]:

						####### Calculating probability of current state given previous state
						c = self.viterbi_cache_2[i-1][1][k][0]*self.transition_matrix[(k,token)]
						if maxx == None or c > maxx:
							maxx = c
							guess = k
					maxx *= self.emission_matrix[(word,token)]


					###### Updating the max probability and the previous state which gave the maximum probability
					self.viterbi_cache_2[i][1][token][0] = maxx
					self.viterbi_cache_2[i][1][token][1] = guess


				else:
					maxx = None
					guess = None
					c = None

					###### Traversing over all the prev_prev states
					for j in self.viterbi_cache_2[i-1][1]:

						####### Traversng over all the prev states and taking the maximum probability path
						for k in self.viterbi_cache_2[i-1][1]:

							######## Calculating the probability of this state given previous two states
							c = self.viterbi_cache_2[i-2][1][j][0]*self.transition_matrix_2[(j,k,token)]
							if maxx == None or c > maxx:
								maxx = c
								guess = k
					maxx *= self.emission_matrix[(word,token)]

					###### Updating the max probability and the previous state which gave the maximum probability
					self.viterbi_cache_2[i][1][token][0] = maxx
					self.viterbi_cache_2[i][1][token][1] = guess



		### Finding the final most probable sequence of tags for our sequence of words
		tokens = []
		token = None

		### Finding the sequence with maximum probability and then backtracking to get the most probable sequence of tags
		for i in range(l-1,-1,-1):
			if token == None:
				maxx = None
				guess = None
				for k in self.viterbi_cache_2[i][1]:
					if maxx == None or self.viterbi_cache_2[i][1][k][0] > maxx:
						maxx = self.viterbi_cache_2[i][1][k][0]
						token = self.viterbi_cache_2[i][1][k][1]
						guess = k
				tokens.append(guess)
			else:
				tokens.append(token)
				token = self.viterbi_cache_2[i][1][token][1]
		tokens.reverse()
		return tokens

#===================================================================================================================================================#



# This is three fold cross validation class, Tasks done by this class:-
# 1 - // 3 fold cross validation\\ - This methods divides the complete sentence dataset into 3 equal sets and then evaluates the HMM model 
#     by taking all the 3 sets one by one as test set and the other two as training set for each itteration. This validation evaluates the 
#     Precision, Recall, Accuracy, F1-score and confusion matrix for all the folds. 
#     /////Implemented in - validation()\\\\\

class Three_fold_cross_vali:


	## Initializing all the necessory fields for the task
	sentences = []                     # Stores the complete set of sentences which will be divided into 3 sets
	confusion_matrix = []              # Stores the confusion matrix of all 3 folds			    
	precisions = []                    # Stores the precisions of all the 3 folds
	recalls = []                       # Stores the recalls of all the 3 folds
	f1_scores = []                     # Stores the f1_scores of all the 3 folds



	## Taking the complete dataset of sentences as input with tags and assigning it to the field 'sentences'
	def __init__(self,sents):
		self.sentences = sents


	## Method which carries out the complete validation task of 3 fold cross validation
	def validation(self):


		### Randomly shuffling all the sentences
		random.shuffle(sentences)

		
		### Evaluating number of sentences in each set
		num_sentences = len(sentences)
		sents_per_fold = num_sentences//3
		

		### Applying 3 fold cross validation
		a = 0
		for fold in range(3):
			print()
			print("--------------FOLD " + str(fold+1) + "---------------")
			print()


			#### Dividing into training and testing sets
			start = fold*sents_per_fold
			end = (fold+1)*sents_per_fold
			test_set = sentences[start:1000]
			training_set = sentences[:start] + sentences[end:]


			#### Training the HMM model on training set
			print("Traning HMM model..............")
			Decoder = HMM_POS_Tagging()
			Decoder.train(training_set)
			print("Model Training successful")
			print()


			#### Printing the training statistics
			
			print("-----------Training Stats----------")
			print("Number of sentences - ",Decoder.num_sentences)
			print("Total number of words - ",Decoder.total_num_words)
			print("Total number of tags - ", Decoder.total_num_tags)
			print("Unique words - ",len(Decoder.unique_words))
			print("Unique tags - ", len(Decoder.unique_tags))
			print("Maximum sentence length - ", Decoder.max_sentence_length)
			print("Minimum sentence length - ", Decoder.min_sentence_length)



			#### Initialising confusion matrix of dimension (num_tags X num_tags)
			total_tags = Decoder.unique_tags
			confusion_matrix_x = {}
			for predicted_tag in total_tags:
				temp = {}
				for original_tag in total_tags:
					temp[original_tag] = 0
				confusion_matrix_x[predicted_tag] = temp



			#### Traversing on each test sentence and testing it on the model
			total_predictions = 0
			for test_sent in test_set:


				##### Splitting the sentence and extracting the sequence of words as a string and the original tags for later evaluation 
				x = test_sent.split(" ")
				test_sentence = ""
				original_tags = []
				for i in x:
					if(i.find("_") == -1):
						test_sentence+= " " + i
						original_tags.append("NULL")
					else:
						i = i.split("_")
						test_sentence+= " " +i[0]
						original_tags.append(i[1])


				##### Getting the prediction of tags for the test sentence
				prediction = Decoder.viterbi_bigram(test_sentence[1:])
				

				##### Filling in the confusion matrix by analysing the orginal with the predicted tags
				words = len(original_tags)
				for i in range(words):
					total_predictions += 1
					if((original_tags[i] in total_tags)):
						confusion_matrix_x[prediction[i]][original_tags[i]] += 1


			#### Calculating precision and averge precision
			precision = {}
			average_precision = 0
			len_total_tags = 0
			for predicted_tag in total_tags:
				denom = 0
				for original_tag in total_tags:
					denom+=confusion_matrix_x[predicted_tag][original_tag]
				if(denom!=0):
					precision[predicted_tag] = confusion_matrix_x[predicted_tag][predicted_tag]/denom
				else:
					precision[predicted_tag] = 0

				if(precision[predicted_tag] != 0):
					len_total_tags += 1
				average_precision += precision[predicted_tag]
			average_precision /= len_total_tags



			#### Calculating recall and average recall
			recall = {}
			average_recall = 0
			len_total_tags = 0
			for predicted_tag in total_tags:
				denom = 0
				for original_tag in total_tags:
					denom+=confusion_matrix_x[original_tag][predicted_tag]
				if(denom!=0):
					recall[predicted_tag] = confusion_matrix_x[predicted_tag][predicted_tag]/denom
				else:
					recall[predicted_tag] = 0

				if(recall[predicted_tag] != 0):
					len_total_tags += 1
				average_recall += recall[predicted_tag]
			average_recall /= len_total_tags


			#### Calculating f1 score
			f1_score = {}
			average_f1_score = 0
			len_total_tags = 0
			for tag in total_tags:

				if(precision[tag] !=0 and recall[tag]!=0):
					score = (2*precision[tag]*recall[tag])/(precision[tag] + recall[tag])
				else:
					score = 0
				f1_score[tag] = score

				average_f1_score += f1_score[tag]
				if(f1_score[tag] != 0):
					len_total_tags += 1
			average_f1_score /= len_total_tags



			#### Calculating accuracy
			numerator = 0
			for tag in total_tags:
				numerator += confusion_matrix_x[tag][tag]
			accuracy = numerator/total_predictions


			##### Calculating word types tagged incorrectly
			arr = []
			for tag1 in total_tags:
				summ = 0
				for tag2 in total_tags:
					summ += confusion_matrix_x[tag2][tag1]
				arr.append([tag1,summ - confusion_matrix_x[tag1][tag1]])
			arr = sorted(arr, key = lambda x : x[1], reverse = True)
			top_5_incorrectly_predicted = []
			for i in range(5):
				top_5_incorrectly_predicted.append(arr[i][0])


			##### Making confusion matrix subset to display
			tss = ["NN", "IN", "JJ", "NNS", "AT"]
			a = []
			count1 = 5
			for i in tss:
				if(count1==0):
					break
				count1-=1
				temp = [i]
				count2 = 5
				for j in tss:
					if(count2 == 0):
						break
					count2-=1
					temp.append(confusion_matrix_x[i][j])
				a.append(temp)

			row = []
			count1 = 5
			for i in tss:
				if(count1==0):
					break
				count1 -=1
				row.append(i)



			#### Printing out the validation statistics for the ith fold
			print()
			print()
			print("------2nd order Hidden Markov Model with Vitirbi algorithm------")
			print()
			print("Accuracy = ",accuracy)
			print()
			print("F1_score of NN, JJ, VBD = ", str([f1_score["NN"],f1_score["JJ"],f1_score["VBD"]]))
			print("Average F1 score over all tags = ", average_f1_score)
			print("Number of F1_scores = ", len(f1_score))
			print()
			print("precision of NN, JJ, VBD = ",str([precision["NN"],precision["JJ"],precision["VBD"]]))
			print("Average precision over all tags = ",average_precision)
			print("Number of precisions = ", len(precision))
			print()
			print("Recall of NN, JJ, VBD = " + str([recall["NN"], recall["JJ"], recall["VBD"]]))
			print("Average recall over all tags = ",average_recall)
			print("Number of recalls = ", len(recall))
			print()
			print("Confusion matrix dimensions (row, column) : " + str([len(confusion_matrix_x),len(confusion_matrix_x["NN"])]))
			print("Subset of confusion matrix:-")
			row1 = ""
			for i in row:
				row1 += "\t" + str(i)
			print(row1)

			for i in a:
				row = ""
				for j in i:
					row += str(j) + "\t"
				print(row)
			print()
			print("Top 5 most incorrectly predicted word types =", str(top_5_incorrectly_predicted))
			

			#### Appending the stats to the fields
			self.confusion_matrix.append(confusion_matrix_x)
			self.precisions.append(precision)
			self.recalls.append(recall)
			self.f1_scores.append(f1_score)
			break


#===================================================================================================================================================#


#################### Driver Code ####################


# Reading the Brown_train.txt file
text = ""
try:
	fp = codecs.open("Brown_train.txt", 'r', encoding = 'utf-8', errors = 'ignore')
	text = fp.read()
except:
	print("File not found")
	text = "File not found"
	exit()
print()
print("----------File Read Successfully---------")
print()



# Splitting sentences by "\n"
sentences = text.split("\n")



# Applying 3 fold cross validation on our complete sentences
Three_fold = Three_fold_cross_vali(sentences)
Three_fold.validation()









