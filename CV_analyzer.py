from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)

import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.core import Settings, VectorStoreIndex  
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import shutil
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
import os
from typing import List, Optional
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, root_validator, field_validator
from dotenv import load_dotenv
load_dotenv()
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai
import numpy as np
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

#Set your API keys in a secret.toml file
openai.api_key = st.secrets["OPENAI_API_KEY"]
LLAMA_CLOUD_API_KEY = st.secrets["LLAMA_CLOUD_API_KEY_2"]

# Pydantic model for extracting education
class Education(BaseModel):
    institution: Optional[str] = Field(None, description="The name of the educational institution")
    degree: Optional[str] = Field(None, description="The degree or qualification earned")
    graduation_date: Optional[str] = Field(None, description="The graduation date (e.g., 'YYYY-MM')")
    details: Optional[List[str]] = Field(
        None, description="Additional details about the education (e.g., coursework, achievements)"
    )

    @field_validator('details', mode='before')
    def validate_details(cls, v):
        if isinstance(v, str) and v.lower() == 'n/a':
            return []
        elif not isinstance(v, list):
            return []
        return v

# Pydantic model for extracting experience
class Experience(BaseModel):
    company: Optional[str] = Field(None, description="The name of the company or organization")
    location: Optional[str] = Field(None, description="The location of the company or organization")
    role: Optional[str] = Field(None, description="The role or job title held by the candidate")
    start_date: Optional[str] = Field(None, description="The start date of the job (e.g., 'YYYY-MM')")
    end_date: Optional[str] = Field(None, description="The end date of the job or 'Present' if ongoing (e.g., 'MM-YYYY')")
    responsibilities: Optional[List[str]] = Field(
        None, description="A list of responsibilities and tasks handled during the job"
    )

    @field_validator('responsibilities', mode='before')
    def validate_responsibilities(cls, v):
        if isinstance(v, str) and v.lower() == 'n/a':
            return []
        elif not isinstance(v, list):
            return []
        return v
# Main class ensapsulating education and epxerience classes with other information
class Candidate(BaseModel):
    name: Optional[str] = Field(None, description="The full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="The email of the candidate")
    age: Optional[int] = Field(
        None,
        description="The age of the candidate."
    )
    skills: Optional[List[str]] = Field(
        None, description="A list of high-level skills possessed by the candidate."
    )
    experience: Optional[List[Experience]] = Field(
        None, description="A list of experiences detailing previous jobs, roles, and responsibilities"
    )
    education: Optional[List[Education]] = Field(
        None, description="A list of educational qualifications of the candidate including degrees, institutions studied in, and dates of start and end."
    )

    @root_validator(pre=True)
    def handle_invalid_values(cls, values):
        for key, value in values.items():
            if isinstance(value, str) and value.lower() in {'n/a', 'none', ''}:
                values[key] = None
        return values

# Class for analyzing the CV contents
class CvAnalyzer:
    def __init__(self, file_path, llm_option, embedding_option):
        self.file_path = file_path
        self.llm_option = llm_option
        self.embedding_option = embedding_option
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._configure_settings()
        self.query_engine = self._create_query_engine(file_path)

    # Function for extracting the data as per the pydantic models
    def extract_candidate_data(self) -> Candidate:
        """
        Extracts candidate data from the resume.

        Returns:
            Candidate: The extracted candidate data.
        """
        print(f"Extracting CV data. LLM: {self.llm_option}")
        # Output Schema
        output_schema = Candidate.model_json_schema()

        # Prompt
        prompt = f"""
                You are expert in analyzing resumes. Use the following JSON schema describing the information I need to extract from the resume.  
                Please extract name, email, age, skills, education, and experience from the CV in the format defined in the following JSON schema:
                ```json
                {output_schema}
                ```json
                Strictly follow this schema to extract the information. Under no circumstances, change the key names of this schema.
                The information against the required fields may not be explicitly mentioned and may be sparsely located. Extract the information by carefully analyzing the resume. 
                Provide the result in a structured JSON format. Please remove any ```json ``` characters from the output. Do not make up any information. If the information for a field is not available, simply output 'n/a'.
                """
        try:
            response = self.query_engine.query(prompt)

            if not response or not response.response:
                raise ValueError("Query engine returned an empty response.")

            # Validate JSON response against the Candidate model
            return Candidate.model_validate_json(response.response)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")  # Log the error for debugging
            raise ValueError("Failed to extract insights. Please ensure the resume and query engine are properly configured.")

    # Function for computing embeddings based on the selected embedding model. These could be CV embeddings, skill embeddings, or job embeddings
    def _get_embedding(self, texts: List[str], model: str) -> torch.Tensor:
        if model.startswith("text-embedding-"):
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            response = client.embeddings.create(input=texts, model=model)
            embeddings = [torch.tensor(item.embedding) for item in response.data]
        elif model == "BAAI/bge-small-en-v1.5":
            tokenizer = AutoTokenizer.from_pretrained(model)
            hf_model = AutoModel.from_pretrained(model).to(self.device)

            embeddings = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = hf_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu())
        else:
            raise ValueError(f"Unsupported embedding model: {model}")

        return torch.stack(embeddings)

    #Compute skill scores based on their semantic similarity (Cosine similarity) with the CV contents
    def compute_skill_scores(self, skills: list[str]) -> dict:
        """
        Compute semantic weightage scores for each skill based on the resume content

        Parameters:
        - skills (list of str): A list of skills to evaluate.

        Returns:
        - dict: A dictionary mapping each skill to a score 
        """
        # Extract resume content and compute its embedding
        resume_content = self._extract_resume_content()

        # Compute embeddings for all skills at once
        skill_embeddings = self._get_embedding(skills, model=self.embedding_model.model_name)

        # Compute raw similarity scores and semantic frequency for each skill
        raw_scores = {}
        # frequency_scores = {}
        for skill, skill_embedding in zip(skills, skill_embeddings):
            # Compute semantic similarity with the entire resume
            similarity = self._cosine_similarity(
                self._get_embedding([resume_content], model=self.embedding_model.model_name)[0],
                skill_embedding
            )
            raw_scores[skill] = similarity
        return raw_scores

    # Extract all the contents from a CV
    def _extract_resume_content(self) -> str:
        """
        Extracts and returns the full text of the resume from the query engine.

        Returns:
        - str: The full text of the resume.
        """
        documents = self.query_engine.retrieve("Extract full resume content")
        return " ".join([doc.text for doc in documents])

    #Function to compute the Cosine similarity of skills with the CV contents
    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.

        Parameters:
        - vec1 (np.ndarray): First vector.
        - vec2 (np.ndarray): Second vector.

        Returns:
        - float: Cosine similarity score.
        """
        vec1, vec2 = vec1.to(self.device), vec2.to(self.device)
        return (torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))).item()

    # Function to create a query engine for parsing and job retrieval
    def _create_query_engine(self, file_path: str):
        """
        Creates a query engine from a file path, handling different LLM configurations.

        Args:
            file_path (str): The path to the file.

        Returns:
            The created query engine.
        """
        parsing_instructions = (
            "This document is a resume. Extract each section separately. The sections may include "
            "personal and contact information, education, skills, experience, publications, etc."
        )

        # Parser
        parser = LlamaParse(
            result_type="markdown",
            parsing_instructions=parsing_instructions,
            auto_mode=True,
            auto_mode_trigger_on_image_in_page=True,
            auto_mode_trigger_on_table_in_page=True,
            api_key=LLAMA_CLOUD_API_KEY,
            verbose=True,
        )
        file_extractor = {".pdf": parser}

        # Reader
        documents = SimpleDirectoryReader(
            input_files=[file_path], file_extractor=file_extractor
        ).load_data()

        # Vector index creation
        index = VectorStoreIndex.from_documents(documents, embed_model=self.embedding_model)
        query_engine = index.as_query_engine()
        print("Query engine initialized successfully.")
        return query_engine
    
    # Function to configure model settings
    def _configure_settings(self):
        """
        Configure the LLM and embedding model based on user selections.
        """
        # Determine the device based on CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA is available. Using GPU.")
        else:
            device = "cpu"
            print("CUDA is not available. Using CPU.")

        # Configure the LLM
        if self.llm_option == "gpt-4o":
            llm = OpenAI(model="gpt-4o", temperature=0.0)
        elif self.llm_option == "gpt-4o-mini":
            llm = Ollama(model="gpt-4o-mini", temperature = 0)
        elif self.llm_option == "llama3:70b-instruct-q4_0":
            llm = Ollama(model="llama3:70b-instruct-q4_0", temperature = 0, request_timeout=180.0, device=device)
        elif self.llm_option == "mistral:latest":
            llm = Ollama(model="mistral:latest", temperature = 0, request_timeout=180.0, device=device)
        elif self.llm_option == "llama3.3:latest":
            llm = Ollama(model="llama3.3:latest", temperature = 0, request_timeout=180.0, device=device)
        else:
            raise ValueError(f"Unsupported LLM option: {self.llm_option}")

        # Configure the embedding model
        if self.embedding_option.startswith("text-embedding-"):
            embed_model = OpenAIEmbedding(model=self.embedding_option)
        elif self.embedding_option == "BAAI/bge-small-en-v1.5":
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_option}")

        # Set the models in Settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        self.llm = llm
        self.embedding_model = embed_model
    
    #Function to create an existing job vector dataset or create a new job vector dataset
    def create_or_load_job_index(self, json_file: str, index_folder: str = "job_index_storage", recreate: bool = False):
        """
        Create or load a vector database for jobs using LlamaIndex.

        Args:
        - json_file: Path to job dataset JSON file.
        - index_folder: Folder to save/load the vector index.
        - recreate: Boolean flag indicating whether to recreate the index.

        Returns:
        - VectorStoreIndex: The job vector index.
        """
        if recreate and os.path.exists(index_folder):
            # Delete the existing job index storage
            print(f"Deleting the existing job dataset: {index_folder}...")
            shutil.rmtree(index_folder)
        if not os.path.exists(index_folder):
            print(f"Creating new job vector index with {self.embedding_model.model_name} model...")
            with open(json_file, "r") as f:
                job_data = json.load(f)
            # Convert job descriptions to Document objects by serializing all fields dynamically
            documents = []
            for job in job_data["jobs"]:
                job_text = "\n".join([f"{key.capitalize()}: {value}" for key, value in job.items()])
                documents.append(Document(text=job_text))
            # Create the vector index directly from documents
            index = VectorStoreIndex.from_documents(documents, embed_model=self.embedding_model)
            # Save index to disk
            index.storage_context.persist(persist_dir=index_folder)
            return index
        else:
            print(f"Loading existing job index from {index_folder}...")
            storage_context = StorageContext.from_defaults(persist_dir=index_folder)
            return load_index_from_storage(storage_context)

    #Function to query job dataset to fetch the top k matching jobs according to the given education, skills, and experience. 
    def query_jobs(self, education, skills, experience, index, top_k=3):
        """
        Query the vector database for jobs matching the resume.

        Args:
        - education: List of educational qualifications.
        - skills: List of skills.
        - experience: List of experiences.
        - index: Job vector database index.
        - top_k: Number of top results to return.

        Returns:
        - List of job matches.
        """
        print(f"Fetching job suggestions.(LLM: {self.llm.model}, embed_model: {self.embedding_option})")
        query = f"Education: {', '.join(education)}; Skills: {', '.join(skills)}; Experience: {', '.join(experience)}"
        # Use retriever with appropriate model
        retriever = index.as_retriever(similarity_top_k=top_k)
        matches = retriever.retrieve(query)
        return matches

if __name__ == "__main__":
    pass


