import argparse
import requests

WATSON_ENDPOINT = ""

def predict_helpfulness(args):
	"""
	Calls Watson ML API
	"""
	params = {}

	res = requests.post(url=WATSON_ENDPOINT, params=params)

	#TODO: output response in a meaningful way

def main():
	parser = argparse.ArgumentParser(description='Estimate the probability that a review is helpful.')
	parser.add_argument('--review', type=str, required=True, help='Text of review you wish to analyze')
	parser.set_defaults(cmd='predict_helpfulness')

	args = parser.parse_args()

if __name__ == "__main__": main()