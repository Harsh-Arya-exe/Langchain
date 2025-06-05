from typing import TypedDict

class Student(TypedDict):

    name: str
    roll: int 

new_student: Student = {'name': 'John Doe', 'roll': 32}

print(new_student)

