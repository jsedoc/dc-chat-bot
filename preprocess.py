import json
import re

TITLE = 'title'
POST = 'post'
COMMENT = 'comment'

def remove_punctuation(line):
	line = re.sub('[?"]','',line)
	return line

def clean(line):
	line = line.replace('\s+',' ').strip()
	line = line.replace('\n',' ').strip()
	line = remove_punctuation(line)
	return line

def read_lines(filename):
        return open(filename,encoding='utf-8').read().split('\n')[:-1]

def create_data_set(lines):
	contexts = []
	responses = []
	len_post = 0
	len_comment = 0
	for line in lines:
		record = json.loads(line)			
		title = record[TITLE].strip()
		title = clean(title)
		post = record[POST].strip()
		post = clean(post)
		comment = record[COMMENT].strip()
		comment = clean(comment)	
		len_post += len(title + post)
		len_comment += len(comment)
		contexts.append(title + post)
		responses.append(comment)
	avg_post = len_post/len(contexts)
	avg_comment = len_comment/len(responses)
	return (contexts,responses)
 
def preprocess_data(filename):		
	lines = read_lines(filename)
	return create_data_set(lines)
	
def save_data(contexts,responses,train_set_path,train_set_target_path,test_set_path,test_set_target_path,test_set_percent):
	test_set_size = int(len(contexts)*test_set_percent)
	train_set_size = len(contexts) - test_set_size
	training_set = contexts[:train_set_size]
	test_set = contexts[train_set_size:len(contexts)]
	training_set_target = responses[:len(training_set)]
	test_set_target = responses[train_set_size:len(responses)]
	save_in_file(training_set,train_set_path)
	save_in_file(training_set_target,train_set_target_path)
	save_in_file(test_set,test_set_path)
	save_in_file(test_set_target,test_set_target_path)

def save_in_file(data_set,filename):
	with open(filename,'w', encoding='utf-8') as data_file:
		for line in data_set:
			data_file.write(line)
			data_file.write('\n')

if __name__ == "__main__":
	filename='data/GamingComments.json'
	train_set_path = 'gaming_train_set'
	train_set_target_path = 'gaming_train_set_target'
	test_set_path = 'gaming_test_set'
	test_set_target_path = 'gaming_test_set_target'
	test_set_percent = 0.2
	(contexts,responses) = preprocess_data(filename)
	save_data(contexts,responses,train_set_path,train_set_target_path,test_set_path,test_set_target_path,test_set_percent)

