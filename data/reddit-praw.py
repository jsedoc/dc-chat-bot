import praw
import sys
from praw.models import MoreComments


LIMIT_POSTS = 3000
SUBREDDIT_TOPIC = "IAmA"

def initializeClass(client_id, client_secret,user_agent):
	return praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)


def extract_single_conversation(top_level_comment):
	sing_conv = []
	sing_conv.append("A : " +top_level_comment.body)
	# print(sing_conv)
	second_level_comment = top_level_comment.replies
	second_level_comment.replace_more(limit=0)
	if(second_level_comment is not None and len(second_level_comment) > 0):
		# if isinstance(second_level_comment[0], MoreComments):
		# 	return sing_conv
		# second_level_comment.replace_more(limit=0)
		sing_conv.append("B : "  + second_level_comment[0].body)
		third_level_comment = second_level_comment[0].replies
		third_level_comment.replace_more(limit=0)
		if(third_level_comment is not None and len(third_level_comment) > 0):
			sing_conv.append("A : "  + third_level_comment[0].body)
			fourth_level_comment = third_level_comment[0].replies
			fourth_level_comment.replace_more(limit=0)
			if(fourth_level_comment is not None and len(fourth_level_comment) > 0):
				sing_conv.append("B : "  + fourth_level_comment[0].body)
				fifth_level_comment = fourth_level_comment[0].replies
				fifth_level_comment.replace_more(limit=0)
				if(fifth_level_comment is not None and len(fifth_level_comment) > 0):
					sing_conv.append("A : "  + fifth_level_comment[0].body)
					sixth_level_comment = fifth_level_comment[0].replies
					sixth_level_comment.replace_more(limit=0)
					if(sixth_level_comment is not None and len(sixth_level_comment) > 0):
						sing_conv.append("B : "  + sixth_level_comment[0].body)
						seventh_level_comment = sixth_level_comment[0].replies
						seventh_level_comment.replace_more(limit=0)
						if(seventh_level_comment is not None and len(seventh_level_comment) > 0):
							sing_conv.append("A : "  + seventh_level_comment[0].body)
							eigth_level_comment = seventh_level_comment[0].replies
							eigth_level_comment.replace_more(limit=0)
							if(eigth_level_comment is not None and len(eigth_level_comment) > 0):
								sing_conv.append("B : "  + eigth_level_comment[0].body)

	cleaned_sing_conv = []

	for conv in sing_conv:
		conv = conv.replace('\n', ' ').replace('\r', '')
		cleaned_sing_conv.append(conv)

	return cleaned_sing_conv




def extract_conversations(submission):
	convs = []
	for top_level_comment in submission.comments:
		if isinstance(top_level_comment, MoreComments):
			continue
		conversation = extract_single_conversation(top_level_comment)
		convs.append(conversation)
	return convs

if __name__ == "__main__":

	reddit = initializeClass(sys.argv[1],sys.argv[2],sys.argv[3])
	subreddit = reddit.subreddit(SUBREDDIT_TOPIC)


	# for submission in subreddit.top(limit=LIMIT_POSTS):
	# 	# print(submission.title)
	# 	# sub_id = submission.id
	# 	output_list = []
	# 	if(submission.link_flair_css_class != 'actor' and submission.link_flair_css_class != 'gaming'):
	# 		continue
	# 	print(submission.title)
	# 	print(submission.link_flair_css_class)
	# 	conversations_list = extract_conversations(submission) #Returns multiple conversations on a single post
	# 	output_list.extend(conversations_list)
	# 	# print(output_list)
	
	sub_list = []

	for f in subreddit.search('flair:"actor"',limit=None):
		sub_list.append(f)

	for f in subreddit.search('flair:"gaming"',limit=None):
		sub_list.append(f)	


	f= open('conversations_reddit_top', 'w')


	for submission in sub_list:
		output_list = []
		# if(submission.link_flair_css_class != 'actor' and submission.link_flair_css_class != 'gaming'):
		# 	continue
		print(submission.title)
		print(submission.link_flair_css_class)
		conversations_list = extract_conversations(submission) #Returns multiple conversations on a single post
		output_list.extend(conversations_list)
		# print(output_list)	


		for post in output_list:
			for conv in post:
				f.write(conv + "\n")
			f.write("\n============================================================================\n\n")
	
	f.close()


	# MAKE UR OWN SUBMISSION 
	#submission = reddit.submission(url= ".....") #Method 1
	#submission = reddit.submission(id= ".....") #Method 2