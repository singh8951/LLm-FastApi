from fastapi import FastAPI
from services.llm_service import load_model
from services.chat_service import ChatService
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model, tokenizer = load_model()

# Initialize chat service
chat_service = ChatService(model, tokenizer)


class ChatRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate/")
async def generate(request: ChatRequest):
    return chat_service.generate_response(request.prompt, request.max_length)


@app.get("/health/")
async def health_check():
    return {"status": "healthy"}
