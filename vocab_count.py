import nltk
import re

nltk.download()

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

def count(filename):
	lines = read_lines(filename)
	print('Lines Read!!')
	full_text = ''
	full_text = full_text.join(lines)
	full_text = clean(full_text)	
	print('Text cleaned!!!')
	words = nltk.word_tokenize(full_text)
	fdist = nltk.FreqDist(words)
	return fdist

if __name__ == '__main__':
	fdist = count('data/ood/full_set')
	count = 0
	for word,frequency in fdist.most_common():
		if (frequency > 1):
			count+=1
		else:
			break
	print(count)
	print(len(fdist))
