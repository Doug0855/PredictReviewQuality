import argparse
import requests
import json
import numpy as np


# Important!
# pip3 install watson-developer-cloud>=2.0.1
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
  import Features, EntitiesOptions, KeywordsOptions
import urllib3




def predict_helpfulness(review):
	"""
	Calls Watson ML API
	"""
	analysis = getAnalysis(review)
	response_scoring = getWatsonLabel(analysis)
	response = json.loads(response_scoring.text)['values'][0]
	prediction = np.around(np.array(response[8], float), 5)
	label = int(response[9])

	output = ["This review is not useful with probability {} %.".format(max(prediction)*100), 
			  "This review is useful with probability {} %.".format(max(prediction)*100)]
	print(output[label])

def getAnalysis(review):
	natural_language_understanding = NaturalLanguageUnderstandingV1(
	  username='10c1b77b-653a-490a-b22e-7ae8367e96ca',
	  password='zLeo2JvxWvRb',
	  version='2018-03-16')
	response = natural_language_understanding.analyze(
	  text=review,
	  features=Features(
	    entities=EntitiesOptions(
	      emotion=True,
	      sentiment=True,
	      limit=2),
	    keywords=KeywordsOptions(
	      emotion=True,
	      sentiment=True,
	      limit=2))).get_result()
	keywords = response["keywords"]
	numKeywords = len(keywords)
	if numKeywords == 0:
	  print("no keywords")
	  return np.zeros(6)
	sentiments = np.array([keyword["sentiment"]["score"] * keyword["relevance"] for keyword in keywords])
	totalSentiment = np.sum(sentiments) / numKeywords

	emotionNames = ['sadness', 'joy', 'fear', 'disgust', 'anger']
	emotions = np.array([np.array([keyword["emotion"][name] for name in emotionNames]) for keyword in keywords])
	totalEmotions = np.sum(emotions, 0) / numKeywords

	features = np.insert(totalEmotions, 0, totalSentiment, axis=0)
	return features

def getWatsonLabel(features):
	# retrieve your wml_service_credentials_username, wml_service_credentials_password, and wml_service_credentials_url from the
	# Service credentials associated with your IBM Cloud Watson Machine Learning Service instance

	wml_credentials={
		"url": 'https://us-south.ml.cloud.ibm.com',
		"username": "80f4b022-d0c7-4a81-a15c-6dbca4a4a6c5",
		"password": "c105b757-0c2e-46bc-bd27-03a64f91d8eb"
	}

	headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'], password=wml_credentials['password']))
	url = '{}/v3/identity/token'.format(wml_credentials['url'])
	response = requests.get(url, headers=headers)
	mltoken = json.loads(response.text).get('token')

	header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

	# NOTE: manually define and pass the array(s) of values to be scored in the next line
	featureTitles = ["COLUMN1", "COLUMN2", "COLUMN3", "COLUMN4", "COLUMN5", "COLUMN6"]
	payload_scoring = {"fields": featureTitles, "values": [features.tolist()]}
	print(payload_scoring)

	response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/v3/wml_instances/c2944031-7793-4406-9b19-a26bd32fec72/deployments/f0730bbc-8048-4752-abc2-8bb9c649f3f9/online', json=payload_scoring, headers=header)
	return response_scoring

def main():
	parser = argparse.ArgumentParser(description='Estimate the probability that a review is helpful.')
	parser.add_argument('--review', type=str, required=True, help='Text of review you wish to analyze')
	parser.set_defaults(cmd='predict_helpfulness')

	args = parser.parse_args()
	predict_helpfulness(args.review)

if __name__ == "__main__": main()