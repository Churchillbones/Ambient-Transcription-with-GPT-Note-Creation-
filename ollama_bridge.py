# Simple API bridge for Ollama to use with the transcription app
from flask import Flask, request, jsonify
import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ollama_bridge")

app = Flask(__name__)

# Model name mapping (Windows filename to Ollama model name)
MODEL_MAPPING = {
    "gemma3-4b": "gemma3:4b",
    "deepseek-r1-14b": "deepseek-r1:14b",
    "llama3": "llama3:latest",
    "phi4": "phi4:latest"
}

@app.route('/generate_note', methods=['POST'])
def generate_note():
    """Generate a note using Ollama API."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        # Get model name from request
        requested_model = data.get('model', 'gemma3-4b')  # Default to gemma3-4b
        
        # Map directory name to Ollama model name
        if requested_model in MODEL_MAPPING:
            model = MODEL_MAPPING[requested_model]
        else:
            # Try to convert hyphens to colons as fallback
            model = requested_model.replace('-', ':')
            logger.info(f"Model mapping not found, converted {requested_model} to {model}")
        
        logger.info(f"Received model name: {requested_model}, using Ollama model: {model}")
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Format the prompt for Ollama
        system_message = (
            "You are a medical transcription assistant specialized in formatting "
            "medical notes based on transcribed audio. Format the transcription into "
            "a proper medical note according to the instructions."
        )
        
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system_message,
                "stream": False
            },
            timeout=120  # Increased timeout to 2 minutes
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return jsonify({"error": f"Ollama API error: {response.status_code}"}), 500
        
        result = response.json()
        note = result.get('response', '')
        
        return jsonify({"note": note})
    
    except requests.exceptions.Timeout:
        logger.error("Timeout while waiting for Ollama API response")
        return jsonify({"error": "Ollama API timeout - model may be taking too long to respond"}), 504
    
    except Exception as e:
        logger.error(f"Error in generate_note: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Ollama API bridge on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)