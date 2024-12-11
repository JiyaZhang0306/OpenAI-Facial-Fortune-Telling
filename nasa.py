from openai import AzureOpenAI
import os
import requests
import json
from PIL import Image

# NASA ROVER 
with open('nasakey.txt', 'r') as nasa_file:
	nasa_key = nasa_file.read().strip()


def get_rover_img_url(rover_date): 
	url = f"https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?earth_date={rover_date}&api_key={nasa_key}"
	response = requests.get(url)
	data = response.json()
	rover_img_url = data["photos"][0]["img_src"]
	return rover_img_url

def get_dalle_image(rover_img_url):
	bot_response = gpt_describe(rover_img_url)
	new_dalle_img = make_dalle_img(bot_response)
	return new_dalle_img

# DALLE
client = AzureOpenAI(
	api_key = os.getenv("AZURE_KEY"),
	api_version = "2024-10-21",
	azure_endpoint = os.getenv("AZURE_ENDPOINT")
)

dalle_folder= 'static'
dalle_image_dir = os.path.join(os.curdir, dalle_folder)

if not os.path.isdir(dalle_image_dir):
	os.mkdir(dalle_image_dir)

def make_dalle_img(bot_response): 
	dalle_prompt = bot_response

	dalle_result = client.images.generate(
		model = "dalle3", 
		prompt = dalle_prompt,
		n=1
		)

	json_response = json.loads(dalle_result.model_dump_json())

	dalle_filename = 'Dalle_Image.png'
	image_path = os.path.join(dalle_image_dir, dalle_filename)
	image_url = json_response['data'][0]['url']

	dalle_gen_image = requests.get(image_url).content


	with open(image_path, 'wb') as file:
		file.write(dalle_gen_image)

	return '/static/'+dalle_filename

def get_dalle_filename():
	return f"/{dalle_folder}/{dalle_filename}"


def gpt_describe(rover_img):
	messages = [
		{
			"role": "system",
			"content": "You are a detailed image analysis system for Mars Rover images. Provide a description of the terrain, textures, and notable features in no more than 500 characters. Based on the environmental conditions observed, design one futuristic speculative survival tool. The tool should directly address challenges suggested by the atmosphere and conditions and be described clearly without using bold text."
		},
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "Analyze the provided image captured by the Mars Rover."},
				{
					"type": "image_url",
					"image_url": {
						"url": rover_img
					}
				}
			]
		}
	]

	response = client.chat.completions.create(
		model="gpt-4o",
		messages=messages,
		tools=functions,
		tool_choice="auto"
	)
	return response.choices[0].message.content


functions = [
	{
		"type": "function", 
		"function": {
			"name": "get_img_url",
			"description": "Gets the image url from the Mars Rover Api",
			"parameters": {
				"type": "object",
				"properties": {
					"rover_date": {
						"type": "string",
						"description": "The date of the Mars Rover image i want to look up"
					},

				},
				"required":["rover_date"]
			}

		}
	}
]