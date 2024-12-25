from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
import os
from typing import List, Optional
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, root_validator, field_validator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai
import numpy as np
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.openai import OpenAI
import streamlit as st


openai.api_key = st.secrets["OPENAI_API_KEY"]
LLAMA_CLOUD_API_KEY = st.secrets["LLAMA_CLOUD_API_KEY_2"]

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

class CvAnalyzer:
    def __init__(self, file_path):
        self._configure_settings()
        self.query_engine = self._create_query_engine(file_path)
        self.embedding_model = Settings.embed_model
        self.llm = Settings.llm 
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def extract_candidate_data(self) -> Candidate:
        """
        Extracts candidate data from the resume.

        Returns:
            Candidate: The extracted candidate data.
        """
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

            # Clean the response to remove extraneous formatting
            cleaned_response = response.response.strip().strip("```json").strip("```")

            # Validate JSON response against the Candidate model
            return Candidate.model_validate_json(cleaned_response)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")  # Log the error for debugging
            raise ValueError("Failed to extract insights. Please ensure the resume and query engine are properly configured.")

    def _get_embedding(self, texts: list[str], model) -> np.ndarray:
        """
        Compute embeddings for a list of texts using OpenAI.

        Parameters:
        - texts (list of str): The texts to compute embeddings for.
        - model (str): The model to use for embeddings. Defaults to "text-embedding-ada-002".

        Returns:
        - np.ndarray: A NumPy array containing embedding vectors for all provided texts.
        """
        from openai import OpenAI

        client = OpenAI(api_key = openai.api_key)
        response = client.embeddings.create(input=texts, model=model)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

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

    def _extract_resume_content(self) -> str:
        """
        Extracts and returns the full text of the resume from the query engine.

        Returns:
        - str: The full text of the resume.
        """
        documents = self.query_engine.retrieve("Extract full resume content")
        return " ".join([doc.text for doc in documents])

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Parameters:
        - vec1 (np.ndarray): First vector.
        - vec2 (np.ndarray): Second vector.

        Returns:
        - float: Cosine similarity score.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _create_query_engine(self, file_path: str):
        """
        Creates a query engine from a file path.

        Args:
            file_path (str): The path to the file.

        Returns:
            The created query engine.
        """

        parsing_instructions = (
            "This document is a resume. Extract each section separately, including personal and contact information, "
            "education, skills, experience, publications, etc. Ensure that each section is output as a separate document object."
        )

        # Parser
        parser = LlamaParse(
            #result_type="text",  
            result_type = "markdown",
            premium_mode=True,
            parsing_instructions = parsing_instructions,
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

        #print(documents)
        
        # Vector index
        index = VectorStoreIndex.from_documents(documents)
        # Query Engine
        return index.as_query_engine()

    def _configure_settings(self):
        """
        Set the LLM and the embedding model
        """
        llm = OpenAI(model="gpt-4o", temperature = 0.0)
        embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        # Global Settings
        Settings.embed_model = embed_model
        Settings.llm = llm 


    def create_or_load_job_index(self, json_file: str, index_folder: str = "job_index_storage"):
        """
        Create or load a vector database for jobs using LlamaIndex.

        Args:
        - json_file: Path to job dataset JSON file.
        - index_folder: Folder to save/load the vector index.

        Returns:
        - VectorStoreIndex: The job vector index.
        """
        if os.path.exists(index_folder):
            print(f"Loading existing job index from {index_folder}...")
            storage_context = StorageContext.from_defaults(persist_dir=index_folder)
            return load_index_from_storage(storage_context)

        print("Creating new job vector index...")
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
        query = f"Education: {', '.join(education)}; Skills: {', '.join(skills)}; Experience: {', '.join(experience)}"
        
        # Configure the retriever with the desired number of top results
        retriever = index.as_retriever(similarity_top_k=top_k)
        
        # Retrieve the top matching jobs
        matches = retriever.retrieve(query)

        return matches
 
if __name__ == "__main__":
    pass

