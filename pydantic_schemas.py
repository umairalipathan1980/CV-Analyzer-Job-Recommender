##Pydantic models for extracting required information

from typing import List, Optional
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, root_validator, field_validator


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
        description="The age of the candidate. Must be an integer between 18 and 99.",
        ge=18, le=99
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
