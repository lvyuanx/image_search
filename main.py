from app import app
from image_search_engine import warm_up_image_search
import uvicorn

if __name__ == "__main__":
    warm_up_image_search()
    uvicorn.run(app, host="0.0.0.0", port=8000)