from fastapi import FastAPI, BackgroundTasks


app = FastAPI()

@app.get("/")
def get_root() -> dict:
    return {'message': 'hello world!'}

