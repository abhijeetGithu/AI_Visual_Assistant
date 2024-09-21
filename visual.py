import streamlit as st
from io import BytesIO
import scipy
import numpy as np 
from typing import Any
from PIL import Image  # Add this import
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
import sys
from transformers import VitsModel, VitsTokenizer
import torch
# LLaVA Model
model_mmproj_file = "llava-v1.5-7b-mmproj-f16.gguf"
model_file = "llava-v1.5-7b-Q4_K.gguf"

def image_b64encode(img: Image) -> str:
    """ Convert image to a base64 format """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
def model_inference(model: Any, request: str, image: Image) -> str:
    """ Ask model a question """
    image_b64 = image_b64encode(image)
    out_stream = model.create_chat_completion(
      messages = [
          {
              "role": "system",
              "content": "You are an assistant who perfectly describes images."
          },
          {
              "role": "user",
              "content": [
                  {"type": "image_url",
                   "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                  {"type" : "text",
                   "text": request}
              ]
          }
      ],
      stream=True,
      temperature=0.2
    )

    # Get characters from stream
    output = ""
    for r in out_stream:
        data = r["choices"][0]["delta"]
        if "content" in data:
            print(data["content"], end="")
            sys.stdout.flush()
            output += data["content"]

    return output




@st.cache_resource
def load_chat_handler():
    """ Load LLAVA chat handler """
    return Llava15ChatHandler(clip_model_path=model_mmproj_file)

@st.cache_resource
def load_model():
    """ Load model """
    chat_handler = load_chat_handler()
    return Llama(
        model_path=model_file,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1,  # Set to 0 if you don't have a GPU,
        verbose=True,
        logits_all=True,
    )


class TTSModel:
    def __init__(self):
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    def generate_audio(self, text: str) -> bytes:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            data = self.model(**inputs).waveform.cpu().numpy()
        
        sample_rate = 16000
        buffer = BytesIO()
        data_int16 = (data * np.iinfo(np.int16).max).astype(np.int16)
        scipy.io.wavfile.write(buffer, rate=sample_rate, data=data_int16.squeeze())
        return buffer.getbuffer().tobytes()

def load_tts_model():
    """ Load the TTS model """
    return TTSModel()



def st_describe(model: Any, tts: Any, prompt: str, image: Image) -> str:
    """ Describe image with a prompt in browser """
    with st.spinner('Describing the image...'):
        response = model_inference(model, prompt, image)
    st_generate_audio(tts, response)

def st_generate_audio(tts: Any, text: str):
    """ Generate and play the audio"""
    with st.spinner('Generating the audio...'):
        wav_data = tts.generate_audio(text)    
    st_autoplay(wav_data)

def st_autoplay(wav: bytes):
    """ Create an audio control in browser """
    b64 = base64.b64encode(wav).decode()
    md = f"""
         <audio controls autoplay="true">
         <source src="data:audio/mp3;base64,{b64}" type="audio/wav">
         </audio>
         """
    st.markdown(md, unsafe_allow_html=True)



def main():
    """ Main app """
    st.title('LLAVA AI Assistant')

    with st.spinner('Loading the models, please wait'):
        model = load_model()
        tts = load_tts_model()

    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer:
        cam_image = Image.open(img_file_buffer)
        if st.button('Describe the image'):
            st_describe(model, tts, "Please describe the image.", cam_image)
        if st.button('Read the label'):
            st_describe(model, tts, "Read the text on the image. If there is no text, write that the text cannot be found.", cam_image)



if __name__ == "__main__":
    main()