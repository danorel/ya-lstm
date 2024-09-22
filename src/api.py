import torch
import tritonclient.grpc as grpcclient

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.constants import INDEX_TO_CHAR, PAD_CHAR
from src.utils import embedding_from_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = grpcclient.InferenceServerClient(url="localhost:8001")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:8080"}})


@app.route('/infer', methods=['POST'])
def infer():
    data = request.json

    model = data.get('model', 'lstm')
    prompt = data.get('prompt', '')
    sequence_size = int(data.get('sequenceSize', '64'))

    if len(prompt) < sequence_size:
        pad_size = sequence_size - len(prompt)
        padded_prompt = [PAD_CHAR] * pad_size + list(prompt)
    else:
        padded_prompt = list(prompt[-sequence_size:])

    prompt_embedding = embedding_from_prompt(padded_prompt)
    prompt_embedding = prompt_embedding.unsqueeze(0).to(device)

    inputs = [grpcclient.InferInput('input', prompt_embedding.shape, "FP32")]
    inputs[0].set_data_from_numpy(prompt_embedding.numpy())

    outputs = [grpcclient.InferRequestedOutput('output')]

    response = client.infer(model, inputs, outputs=outputs)
    response_output = response.as_numpy('output')

    logits = torch.from_numpy(response_output)
    logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
    char_idx = torch.multinomial(logits_probs, 1).item()
    char = INDEX_TO_CHAR[char_idx]

    return jsonify({ "char": char }), 200

if __name__ == '__main__':
    app.run(debug=True, port=4000)
