# CV Analyzer and Job Recommender

This tool analyzes a CV and extracts key information, scores the extracted skills, and suggests suitable jobs from a job ad database.  

There are two folders: 
1. **OpenAI models**: the code in this folder uses OpenAI's gpt-4o large language model, and text-embedding-3-large embedding model
2. **Mutiple models**: the code in this folder offers to select multiple models including open-source and OpenAI's large language models and embedding models.  

If you have a powerful computing machine with a GPU, you can try the code in **Multiple models** for comapring the performance of multiple models. If you prefer to use only OpenAI models, you need to run the code in **OpenAI models**. In that case you will need OpenAI's API keys. 

The instructions to use the codes in both the folders are present in their respective folders. 

## Features
- Option to select open-source or OpenAI's large language models and embedding models (the folder Multiple models) 
- **CV Parsing**: Parsing CV using LlamaParse. 
- **Information Extraction**: Extracting information from CV using Pydantic models and gpt-4o
- **Skills scoring**: Assigning scores to skills and displaying in a UI
- **Job Recommendations**: Recommding jobs based on the profile from a job dataset

## Overall Process
The following figure shows the overall process. 

<p align="center">
  <img src="images/image.png" alt="My Figure" width="700">
  <br>
  <em>Overall process of parsing a resume, scoring skills, and matching education, experience and skills with a vector database to suggest matching jobs.</em>
</p>
Following is the screenshot of the streamlit app:  

<p align="center">
  <img src="images/appScrshot.png" alt="My Figure" width="700">
  <br>
  <em>A snapshot of the streamlit application</em>
</p>

## How to Use the Code.

Clone the repository:
   ```
   git clone https://github.com/umairalipathan1980/CV-Analyzer-Job-Recommender.git
   cd CV-Analyzer-Job-Recommender
   ```

For using **only OpenAI models**, nagivate to the folder **OpenAI models**. Follow the instructions in readme file.  

For using **additional models** such as llama3.1:latest, navigate to the folder **Multiple models**. From there, you can select a large language model and an embedding model. In this case, you will need to install additional libraries. Please refer to the readme file in that folder.



