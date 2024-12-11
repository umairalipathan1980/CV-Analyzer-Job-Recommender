import streamlit as st
import tempfile
import random
import os
from CV_analyzer import ResumeInsights
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate


class RAGStringQueryEngine(CustomQueryEngine):
    """
    Custom Query Engine for Retrieval-Augmented Generation.
    """
    retriever: BaseRetriever
    llm: OpenAI
    qa_prompt: PromptTemplate

    def custom_query(self, candidate_details: str, retrieved_jobs: str):
        query_str = self.qa_prompt.format(
            query_str=candidate_details, context_str=retrieved_jobs
        )
        response = self.llm.complete(query_str)
        return str(response)


def main():
    st.set_page_config(page_title="CV Analyzer & Job Recommender", page_icon="📄")
    st.title("CV Analyzer & Job Recommender")
    st.write("Upload a CV to extract key information.")

    uploaded_file = st.file_uploader("Select Your CV (PDF)", type="pdf", help="Choose a PDF file up to 5MB")

    if uploaded_file is not None:
        if st.button("Analyze"):
            with st.spinner("Parsing CV... This may take a moment."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    # Initialize ResumeInsights
                    resume_insights = ResumeInsights(temp_file_path)

                    # Extract insights from the resume
                    insights = resume_insights.extract_candidate_data()

                    # Load or create job vector index
                    job_index = resume_insights.create_or_load_job_index(json_file="sample_jobs.json")

                    # Query jobs based on resume data
                    education = [edu.degree for edu in insights.education] if insights.education else []
                    skills = insights.skills or []
                    experience = [exp.role for exp in insights.experience] if insights.experience else []

                    print(f"Education: {education}\n skills:{skills}\n Experience: {experience}")

                    matching_jobs = resume_insights.query_jobs(education, skills, experience, job_index)

                    # Send retrieved nodes to LLM for final output
                    retrieved_context = "\n\n".join([match.node.get_content() for match in matching_jobs])
                    candidate_details = f"Education: {', '.join(education)}; Skills: {', '.join(skills)}; Experience: {', '.join(experience)}"
                    #print(candidate_details)

                    rag_engine = RAGStringQueryEngine(
                        retriever=job_index.as_retriever(),
                        llm=resume_insights.llm,
                        qa_prompt=PromptTemplate(template="""\
                            You are expert in analyzing resumes, based on the following candidate details and job descriptions:
                            Candidate Details:
                            ---------------------
                            {query_str}
                            ---------------------
                            Job Descriptions:
                            ---------------------
                            {context_str}
                            ---------------------
                            Provide a concise list of the matching jobs. For each matching job,  mention job related details such as 
                            company, brief job description, location, employment type, salary range, URL for each suggestion, and a brief explanation of why the job matches the candidate's profile.
                            Be critical in matching profile with the jobs. Thoroughly analyze education, skills, and experience to match jobs. 
                            Do not explain why the candidate's profile does not match with the other jobs. Do not include any summary.
                            Answer: 
                            """)
                    )

                    llm_response = rag_engine.custom_query(
                        candidate_details=candidate_details,
                        retrieved_jobs=retrieved_context
                    )

                    # Display extracted information
                    st.subheader("Extracted Information")
                    st.write(f"**Name:** {insights.name}")
                    st.write(f"**Email:** {insights.email}")
                    st.write(f"**Age:** {insights.age}")
                    display_education(insights.education or [])
                    display_skills(insights.skills or [], resume_insights)
                    display_experience(insights.experience or [])
                    st.subheader("Top Matching Jobs with Explanation")
                    st.markdown(llm_response)
                except Exception as e:
                    st.error(f"Failed to analyze the resume: {str(e)}")

def display_skills(skills: list[str], resume_insights):
    """
    Display skills with their computed weightage scores as progress bars.

    Parameters:
    - skills (list of str): A list of skills to evaluate.
    - resume_insights (ResumeInsights): An instance of ResumeInsights to compute scores.
    """
    if not skills:
        st.warning("No skills found to display.")
        return

    st.subheader("Skills with Relative Strengths")

    # Custom CSS for progress bars
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #4CAF50, #8BC34A);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Compute scores for all skills
    skill_scores = resume_insights.compute_skill_scores(skills)

    # Display each skill with a progress bar
    for skill in skills:
        score = skill_scores.get(skill, 0) / 100  # Normalize score to 0-1 for progress bar

        # Display skill name and progress bar
        st.markdown(f"**{skill}**")
        st.progress(score)

def display_education(education_list):
    """
    Display a list of educational qualifications.

    Parameters:
    - education_list (list of Education): Each item is an instance of the Education model.
    """
    if education_list:
        st.subheader("Education")
        for education in education_list:
            # Access attributes directly since Education is a Pydantic model
            institution = education.institution if education.institution else "Not found"
            degree = education.degree if education.degree else "Not found"
            year = education.graduation_date if education.graduation_date else "Not found"
            details = education.details if education.details else []

            # Format additional details
            formatted_details = ". ".join(details) if details else "No additional details provided."

            # Display education information
            st.markdown(f"**{degree}**, {institution} ({year})")
            st.markdown(f"_Details_: {formatted_details}")

def display_experience(experience_list):
    """
    Display a single-level bulleted list of experiences.

    Parameters:
    - experience_list (list of Experience): Each item is an instance of the Experience model.
    """
    if experience_list:
        st.subheader("Experience")
        for experience in experience_list:
            # Access attributes directly since Experience is a Pydantic model
            job_title = experience.role if experience.role else "Not found"
            company_name = experience.company if experience.company else "Not found"
            location = experience.location if experience.location else "Not found"
            start_date = experience.start_date if experience.start_date else "Not found"
            end_date = experience.end_date if experience.end_date else "Not found"
            responsibilities = experience.responsibilities if experience.responsibilities else ["Not found"]

            # Combine responsibilities into a brief description
            brief_responsibilities = ", ".join(responsibilities)

            # Display formatted output
            st.markdown(
                f"- Worked as **{job_title}** from {start_date} to {end_date} in *{company_name}*, {location}, "
                f"where responsibilities include {brief_responsibilities}."
            )

if __name__ == "__main__":
    main()


