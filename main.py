from app.main import app, create_app
from app.core.config import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, workers=1, log_level="warning", access_log=False)
