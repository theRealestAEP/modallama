import modal
import subprocess
import time
import requests

stub = modal.Stub("ollama-inference")

def setup_ollama():
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    print("Ollama installed.")

ollama_image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .pip_install("ollama", "requests")
    .run_function(setup_ollama)
)

@stub.function(
    gpu="A10G",
    image=ollama_image,
    timeout=300  # 30 minutes timeout
)
def run_inference(prompt, model_name='dolphin-mixtral'):
    import ollama
    
    print("Starting Ollama server...")
    server_process = subprocess.Popen("ollama serve", shell=True, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for _ in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://127.0.0.1:11434/api/version")
            if response.status_code == 200:
                print("Ollama server is ready.")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("Failed to start Ollama server.")
        print(server_process.stderr.read().decode())
        raise Exception("Ollama server failed to start")

    print(f"Pulling model {model_name}...")
    try:
        subprocess.run(f"ollama pull {model_name}", shell=True, check=True, timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model: {e}")
        raise
    except subprocess.TimeoutExpired:
        print("Timeout while pulling the model")
        raise

    print("Starting inference...")
    try:
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        print("Inference completed successfully.")
        return response['message']['content']
    except ollama.ResponseError as e:
        print(f"Ollama ResponseError: {e.error}")
        raise
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
    finally:
        # Stop the Ollama server
        server_process.terminate()
        server_process.wait()

@stub.local_entrypoint()
def main(prompt: str, model: str = 'dolphin-mixtral'):
    result = run_inference.remote(prompt, model)
    print("\nModel output:")
    print(result)

if __name__ == "__main__":
    main()