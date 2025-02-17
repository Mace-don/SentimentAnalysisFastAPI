from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

# Load the emotion analysis model
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
emotion_analyzer = pipeline("text-classification",
                            model="j-hartmann/emotion-english-distilroberta-base",
                            return_all_scores=True,
                            device=device)

# Define request model
class EmotionRequest(BaseModel):
    questions: list[str]
    answers: list[str]

@app.post("/analyze-emotions")
def analyze_emotions(data: EmotionRequest):
    """
    Analyze emotions in answers using a multi-label emotion model.
    """
    try:
        questions = data.questions
        answers = data.answers

        results = []
        for q, a in zip(questions, answers):
            emotions = emotion_analyzer(a)[0]

            result = {
                "question": q,
                "answer": a,
                "emotions": {emotion['label']: round(emotion['score'], 4) for emotion in emotions},
                "dominant_emotion": max(emotions, key=lambda x: x['score'])['label'],
                "confidence": round(max(emotions, key=lambda x: x['score'])['score'], 4)
            }
            results.append(result)

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
