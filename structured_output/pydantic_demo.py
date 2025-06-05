from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = "Jack"
    age: Optional[int] = None  # optional Value : must write none
    email : EmailStr            # Build in Validation: will give error if the value of not appropriate
    cgpa : float = Field(gt=0, lt=10, default=6.0, description='A decimal value representing cgpa')                      # Field : allows >= / <=, default value, and description : it works like Annotated in Typedict


new_student = {'name':'John Doe', 'age':32, 'email':'abc@gmail.com'}

student = Student(**new_student)

print(student)