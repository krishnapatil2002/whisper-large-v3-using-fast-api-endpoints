import os
import whisper
import uvicorn
from fastapi import FastAPI, File, UploadFile
from moviepy.audio.io.AudioFileClip import AudioFileClip
from fastapi.responses import JSONResponse
from transformers import pipeline

audio_mp3 = "audio_temporary_file.mp3"

app = FastAPI()

whisper_model = whisper.load_model("large-v3")

summarizer = pipeline("summarization")

@app.post("/audio")
async def audio(file: UploadFile = File()):
    if "audio" in file.content_type:
        audio_source = "fastapi_upload_audio_source." + file.filename.split('.')[-1]
        audio_content = await file.read()
        with open(audio_source, "wb") as f:
            f.write(audio_content)
        audio = AudioFileClip(audio_source)
        audio.write_audiofile(audio_mp3, codec="mp3")
        
        result = whisper_model.transcribe(audio_mp3, fp16=False)
        
        transcription_text = result["text"]
        summary = summarizer(transcription_text, max_length=150, min_length=30, do_sample=False)

        os.remove(audio_source)
        os.remove(audio_mp3)
        
        return JSONResponse(content={
            "transcription": transcription_text,
            "summary": summary[0]['summary_text'],
            "timestamps": result["segments"]
        })
    else:
        return JSONResponse(content={"error": "This file is not an audio file"}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
