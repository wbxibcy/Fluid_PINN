from pydantic import BaseModel, EmailStr

class RegisterUserRequest(BaseModel):
    user_name: str
    email: EmailStr
    password: str
    role: str

class LoginUserRequest(BaseModel):
    email: EmailStr
    password: str