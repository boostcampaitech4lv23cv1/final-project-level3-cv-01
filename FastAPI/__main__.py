if __name__ == "__main__":
    import uvicorn

    # uvicorn.run("FastAPI.main:app", host="127.0.0.1", port=8001, reload=True)
    uvicorn.run("FastAPI.main:app", reload=True)
