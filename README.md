# CV Analyzer and Job Recommender

This tool analyzes a CV and extracts key information which is then used to suggest suitable jobs from a job ad database. 

## Features

- **CV Parsing**: Parsing CV using LlamaParse.
- **Information Extraction**: Extracting information from CV using Pydantic models and gpt-4o
- **Skills weighting**: Assigning weightage to skills and displaying in a UI
- **Job Recommendations**: Recommding jobs based on the profile.

## How to Use the Code.

1. Clone the repository:
   ```
   git clone https://github.com/umairalipathan1980/CV-Analyzer-Job-Recommender.git
   cd CV-Analyzer-Job-Recommender
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in .env file:
   ```
   OPENAI_API_KEY=your_OPENAI_api_key
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
   ```

4. Run the Streamlit app:
   ```
   python -m streamlit run app.py
   ```


