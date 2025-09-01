# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy
# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):

	# Create a SentimentIntensityAnalyzer object.
	sid_obj = SentimentIntensityAnalyzer()

	# polarity_scores method of SentimentIntensityAnalyzer
	# object gives a sentiment dictionary.
	# which contains pos, neg, neu, and compound scores.
	sentiment_dict = sid_obj.polarity_scores(sentence)
	
	print("Overall sentiment dictionary is : ", sentiment_dict)
	print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
	print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
	print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

	print("Sentence Overall Rated As", end = " ")

	# decide sentiment as positive, negative and neutral
	if sentiment_dict['compound'] >= 0.05 :
		print("Positive")

	elif sentiment_dict['compound'] <= - 0.05 :
		print("Negative")

	else :
		print("Neutral")



# Driver code
if __name__ == "__main__" :

	print("\n1st statement :")
	sentence = "Nisam dobar."

	# function calling
	print(sentence)
	sentiment_scores(sentence)

	print("\n2nd Statement :")
	sentence = "The mobile phone I bought was the worst and very expensive"
	print(sentence)
	sentiment_scores(sentence)

	print("\n3rd Statement :")
	sentence ="VADER is very smart, handsome and funny."
	print(sentence)
	sentiment_scores(sentence)

	print("\n4th Statement :")
	sentence ="VADER is VERY SMART, handsome and funny."
	print(sentence)
	sentiment_scores(sentence)

	print("\n4th Statement :")
	sentence ="A thief kills a policeman."
	print(sentence)
	sentiment_scores(sentence)

	print("\n5th Statement :")
	sentence ="A policeman kills a thief."
	print(sentence)
	sentiment_scores(sentence)

	print("\n5th Statement :")
	sentence ="I am happy."
	print(sentence)
	sentiment_scores(sentence)

	print("\n5th Statement :")
	sentence ="I am HAPPY."
	print(sentence)
	sentiment_scores(sentence)

	print("\n5th Statement :")
	sentence =":-)."
	print(sentence)
	sentiment_scores(sentence)

	print("\n5th Statement :")
	sentence ="You are selfish, but reliable."
	print(sentence)
	sentiment_scores(sentence)

	print(((0.5574+0.293+0.5574)*((0.5574+0.293+0.5574)**2+15))**(-1))
	print(0.5984-0.5574)