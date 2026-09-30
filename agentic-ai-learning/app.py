import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "tinyllama:1.1b-chat",
    "prompt": "What is my name, say it !!",
    "stream": False
    
} #sending the payload prompt to the Ollama API

request = urllib.request.Request(
                    OLLAMA_URL, 
                    data=json.dumps(payload).encode("utf-8"), 
                    headers={"Content-Type": "application/json"},
                    method="POST")
# sending the request to the Ollama API and waiting for a response with a timeout of 180 seconds

with urllib.request.urlopen(request, timeout=180) as response:
    result = json.loads(response.read().decode("utf-8"))
#     for line in response:
#         if line:
#             result = json.loads(line.decode("utf-8"))
#             print(result)
# #receiving the response from the Ollama API and printing it to the console

# #raw response from the Ollama API
# print("\nRaw response from Ollama API:")
# print(json.dumps(result, indent=4))


#converting the json response to a python dictionary and printing the response from the Ollama API to the console
print("\nResponse from Ollama API:")
print(result["response"])