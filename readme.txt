Step 2: Create and Activate a Virtual Environment
Create a virtual environment to manage your dependencies.

bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Step 3: Install Dependencies
Install the required Python packages.

bash
pip install -r requirements.txt
Step 4: Add Model File
Ensure that the pre-trained model file (model_cnn3.h5) is saved in the root directory of the project.

Step 5: Run the API
You can run the FastAPI application using the following command:

bash
python main.py
Alternatively, you can run the API with auto-reload for development purposes:

bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
Usage
To classify the genre of an audio file, send a POST request to the /predict_genre/ endpoint with the file path.

Example POST Request
Request Body
json
{
  "file_path": "path/to/your/audio_file.mp3"
}
Response
json
{
  "predicted_genre": "hiphop"
}
Python Example
You can use the requests library in Python to send a request to the API endpoint.

python
import requests

url = "http://127.0.0.1:8000/predict_genre/"
data = {
    "file_path": "path/to/your/audio_file.mp3"
}

response = requests.post(url, json=data)
print(response.json())
