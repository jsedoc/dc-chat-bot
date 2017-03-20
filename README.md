# dc-chat-bot
Multi domain Chat bot handling context and response independently

Data Set:
We used the reddit dataset with domain tagged. We used post resonses pairs tagged with domains (movies,gaming) and we incuded a out of domain data set with the twitter.
We trained 2 seperate seq2seq models on these 3 domains.
We built a clasification model with tf-idf for first classify a query into a domain and then based on the domain classify ran the seq2seq model for that domain to generate the response.

The data set features are:
Movies data set: 899942
Gaming data set: 374391
Twitter(out of domain) data set: 377265

We picked it up because we wanted to do a domain based chat bot and redidt was the data set with a lot of instances for a lot of domains.

The vocabulary size we used based on the words that occured more than 5 times for each of the data set except the out of the domain data set.
Movies data set vocabulary: 158470
Gaming data set vocabulary: 91935
Twitter (out of domain) vocabulary: 40000

Qualitative Examples:
Good Examples

> hi how are you
Out of domain Predicted!
hello how are you

> can you suggest me a game to play
Gaming Predicted!
The Witcher 0

> which movie is better the ring or logan
Movie Predicted!
Avatar

> shall we go for a movie 
Movie Predicted!
no

> i am going to plat GTA are you interested
Gaming Predicted!
What ' s the new one

> do you know the new FIFA is out
Out of domain Predicted!
honestly haha i know !


Bad Examples:
> why did you suggest so 
Out of domain Predicted!
. . . . . . . . . .

> do you have an plans for dinner 
Out of domain Predicted!
let me know how you decide there

> what about hillary clinton
Out of domain Predicted!
she forgot too

> my joystick has stopped working how do i fix it
Out of domain Predicted!
it is real because i never use a _UNK

> my keyboard has stopped working
Out of domain Predicted!
the phone has been working for the past 0 days

We integrated the chatbot with alexa.

The model was trained with cross-entropy as the optimization function and perplexity was monitored. 

The train and test perplexities for the individual models were:
1) Movies model: train perplexity : 9.452 test perplexity: 21.53
2) Gaming model: train perplexity : 4.56 test perplexity: 34.56
3) Twitter model: train perplexity :12.34 test perlexity: 37.21

