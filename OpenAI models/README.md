1. Install dependencies:
   ```
   pip install llama-index openai pydantic[email] streamlit
   ```

2. Create a folder ".streamlit" in the root directory and create a "secrets.toml" file in it. Set your API keys there as follows:
   ```
   OPENAI_API_KEY = "your_OPENAI_api_key"
   LLAMA_CLOUD_API_KEY = "your_llama_cloud_api_key"
   ```

3. Run the Streamlit app:
   ```
   python -m streamlit run .\job_recommender.py  
   ```
The job dataset is in the folder "job_index_storage". To create a new vector database for modified job dataset, just delete the folder "job_index_storage". A new vector database will be created. 

