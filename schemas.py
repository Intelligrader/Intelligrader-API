from pydantic import BaseModel


class GenerateSAQRequest(BaseModel):
    topic: str = "AP United States History"
    grade_level: str = "College Level"
    max_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    model: str | None = None


class GenerateSAQResponse(BaseModel):
    saq_question: str
    model_used: str | None
    tokens_used: int


class GradeSAQRequest(BaseModel):
    question: str
    student_answer: str
    rubric: str
    max_score: int = 4
    temperature: float = 0.2
    top_p: float = 0.9
    model: str | None = None


class GradeSAQResponse(BaseModel):
    score: int
    max_score: int
    feedback: str
    model_used: str | None
    tokens_used: int