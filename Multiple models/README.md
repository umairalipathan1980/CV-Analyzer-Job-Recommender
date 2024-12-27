## 1. Install dependencies:
   ```
   pip install llama-index openai pydantic[email] streamlit llama-index-llms-ollama llama-index-embeddings-huggingface transformers
   ```
   **Install *torch* for CPU-version (if you do not have a GPU):**
   ```
   pip install torch torchvision torchaudio
   ```
   **Install *torch* for GPU-version (if you have a CUDA-enabled GPU for faster processing). Here, we assume that CUDA toolkit is already installed and configured on your machine.** 
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<CUDA_VERSION>
   ```
   Here **<CUDA_VERSION>** refers to the CUDA version on your PC. For example, on my machine, the CUDA version is 12.6, so I will run the following code:  
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   **Install Ollama from their official website (https://ollama.com/)** 
   
## 2. Set up API keys:
   Create a folder ".streamlit" in the root directory and create a "secrets.toml" file in it. Set your API keys there as follows:
   ```
   OPENAI_API_KEY = "your_OPENAI_api_key"
   LLAMA_CLOUD_API_KEY = "your_llama_cloud_api_key"
   ```
## 3. Run the Streamlit app:
   ```
   python -m streamlit run .\job_recommender.py
   ```
The job dataset is in the folder "job_index_storage". To create a new vector database for modified job dataset, just delete the folder "job_index_storage". A new vector database will be created.


