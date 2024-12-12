from flask import Flask, request, jsonify,render_template
import cv2
import asyncio
from aiohttp import ClientSession, FormData
import numpy as np  # 添加这行代码来导入 numpy 库
from fortuneteller import *


app = Flask(__name__)

# ... (其余函数保持不变，包括read_credentials, detect_faces_async, analyze_faces_async, retry_on_error)
# Semaphore to limit concurrent requests
# semaphore = asyncio.Semaphore(1)

@app.route('/')
def index():
    return render_template('index.html')

# Function to read API credentials
def read_credentials(api_key_file, api_secret_file):
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()
    with open(api_secret_file, 'r') as file:
        api_secret = file.read().strip()
    return api_key, api_secret

# Function to detect faces
async def detect_faces_async(frame, api_key, api_secret, session,semaphore):
    async with semaphore:  # Use semaphore to limit concurrency
        url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        _, img_encoded = cv2.imencode('.jpg', frame)

        # Prepare the request
        form = FormData()
        form.add_field("image_file", img_encoded.tobytes(), filename="frame.jpg", content_type="image/jpeg")
        form.add_field("api_key", api_key)
        form.add_field("api_secret", api_secret)

        try:
            async with session.post(url, data=form, ssl=False) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"[DEBUG] Detect Faces Response: {result}")

                    if 'faces' in result and result['faces']:
                        face_tokens = [face['face_token'] for face in result['faces']]
                        print(f"[DEBUG] Retrieved Face Tokens: {face_tokens}")
                        return face_tokens
                else:
                    error_text = await response.text()
                    print(f"[DEBUG] Detect Faces Error: {response.status} - {error_text}")
        except Exception as e:
            print(f"[DEBUG] Detect Faces Exception: {e}")
    return None

# Function to analyze faces
async def analyze_faces_async(face_tokens, api_key, api_secret, session,semaphore):
    async with semaphore:  # Use semaphore to limit concurrency
        url = "https://api-us.faceplusplus.com/facepp/v3/face/analyze"

        # Prepare the request with only valid attributes
        data = {
            "api_key": api_key,
            "api_secret": api_secret,
            "face_tokens": ",".join(face_tokens),
            "return_attributes": "emotion"  # Valid attributes
        }

        print(f"[DEBUG] Sending Face Tokens for Analysis: {face_tokens}")

        try:
            async with session.post(url, data=data, ssl=False) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"[DEBUG] Analyze Faces Response: {result}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"[DEBUG] Analyze Faces Error: {response.status} - {error_text}")
        except Exception as e:
            print(f"[DEBUG] Analyze Faces Exception: {e}")
    return None


# Retry mechanism with exponential backoff
async def retry_on_error(coro_func, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return await coro_func()  # Properly call the coroutine
        except Exception as e:
            print(f"[DEBUG] Retry ({attempt + 1}/{retries}) after error: {e}")
            await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
    print("[DEBUG] All retries failed.")
    return None

# Flask route to process the video frame
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame part'}), 400

    frame = request.files['frame']
    # Convert image file stream to numpy array
    np_frame = cv2.imdecode(np.frombuffer(frame.read(), np.uint8), cv2.IMREAD_COLOR)

    # Read API credentials
    api_key_file = "api_key.txt"
    api_secret_file = "api_secret.txt"
    api_key, api_secret = read_credentials(api_key_file, api_secret_file)

    # Process the frame
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create a semaphore
    semaphore = asyncio.Semaphore(1)

    async def process():
        async with ClientSession() as session:
            face_tokens = await detect_faces_async(np_frame, api_key, api_secret, session, semaphore)
            if face_tokens:
                analyze_result = await analyze_faces_async(face_tokens, api_key, api_secret, session, semaphore)
                if analyze_result and 'faces' in analyze_result:
                    for face in analyze_result['faces']:
                        emotions = face['attributes'].get('emotion', {})
                        dominant_emotion = max(emotions, key=emotions.get) if emotions else "N/A"

                        # Add a message to ChatGPT
                        emotion_text = (
                            f"The dominant emotion detected is '{dominant_emotion}'. "
                            f"The detailed emotions are: {emotions}. "
                            "Please provide a personalized fortune-telling biscuit based on this information."
                        )
                        messages = [
                        {"role": "system", "content": (
                                 "You are a cool fortune teller. Based on detected emotions from a person's facial expressions, "
                                 "offer a personalized fortune-telling biscuit. Examples:\n"
                                 )},
                        {"role": "user", "content": emotion_text}
                        ]

                        try:
                            response = client.chat.completions.create(
                                model="GPT-4",
                                messages=messages
                            )
                            chat_response = response.choices[0].message.content
                            print("[DEBUG] GPT Response Content:", chat_response)
                            return {
                                "emotion_analysis": analyze_result,
                                "chat_response": chat_response
                            }
                        except Exception as e:
                            print(f"[DEBUG] GPT API Exception: {e}")
                            return {
                                "emotion_analysis": analyze_result,
                                "dominant_emotion": dominant_emotion,
                                "error": "Failed to get ChatGPT response"
                            }

                return {'error': 'No faces detected or no emotion data available'}
            else:
                return {'error': 'No faces detected'}

    try:
        # Run the async process function
        analyze_result = loop.run_until_complete(process())
        return jsonify(analyze_result)
    finally:
        loop.close()

        
if __name__ == "__main__":
    app.run(debug=True)