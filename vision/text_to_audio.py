from openai import OpenAI

client = OpenAI(api_key = "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")

def text_to_audio(text, output_path = "output.mp3"): #takes in a text and returns the output path of the audio
    response = client.audio.speech.create(
        model = "tts-1",
        voice = "alloy",
        input = text
    )
    response.stream_to_file(output_path)
    return output_path
