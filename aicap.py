import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from llama_cpp import Llama
import os
import random
import pysrt  # Library to handle SRT files
from moviepy.editor import VideoFileClip  # For handling video and audio
import subprocess  # For running shell commands
from prompts import STORY, END_PROMPT_FIRST_CHUNK, END_PROMPT_OTHER_CHUNKS, REPLACEMENTS

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Function to calculate the sharpness of a frame
def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Function to extract frames from the video
def get_frames(video_capture, frame_rate=1, sharpness_threshold=20.0):
    frames = []
    count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if count % frame_rate == 0:
            sharpness = calculate_sharpness(frame)
            if sharpness >= sharpness_threshold:
                frames.append((count, frame))
        count += 1
    return frames


def replace_words(text, replacements):
    words = text.split()  # Split the text into words
    replaced_words = [replacements.get(word, word) for word in words]  # Replace words using the dictionary
    return ' '.join(replaced_words)  # Join the words back into a single string

# Set up the Llama model
model_path = "./nous-hermes-2-solar-10.7b.Q5_K_M.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

llm = Llama(
    model_path=model_path,  # Ensure the model file is downloaded
    n_ctx=8192,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=200  # The number of layers to offload to GPU, if you have GPU acceleration available
)

# Directory containing videos
video_directory = './vids/'

# Get the list of video files
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]

# Shuffle the list of video files
random.shuffle(video_files)

# Iterate over the shuffled list of video files
for video_file in video_files:
    if video_file.endswith('.mp4'):
        try:
            captions = []
            video_path = os.path.join(video_directory, video_file)
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            interval_seconds = 10  # Interval in seconds
            frames_per_interval = int(fps * interval_seconds)

            # Extract frames from the video
            frames = get_frames(cap, frame_rate=30, sharpness_threshold=20.0)  # Extract every frame and apply sharpness filter

            # Dictionary to store frame chunks
            frame_chunks = {}

            # Divide frames into chunks
            for i, (frame_idx, frame) in enumerate(frames):
                chunk_idx = frame_idx // frames_per_interval
                if chunk_idx not in frame_chunks:
                    frame_chunks[chunk_idx] = []
                frame_chunks[chunk_idx].append((frame_idx, frame))

            cap.release()

            # Process each chunk
            combined_caption = ""
            for chunk_idx in frame_chunks:
                chunk_frames = frame_chunks[chunk_idx]

                # Analyze and describe each frame in the chunk
                frame_descriptions = []
                for frame_idx, frame in chunk_frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    inputs = processor(pil_image, return_tensors="pt").to(device)
                    outputs = blip_model.generate(**inputs, max_new_tokens=200)
                    description = processor.decode(outputs[0], skip_special_tokens=True)
                    frame_descriptions.append(description + " , ")

                # Combine frame descriptions into a single text
                combined_descriptions = " ".join(frame_descriptions)

                prefix = ""

                replacements = REPLACEMENTS

                if chunk_idx == 0:
                    end = END_PROMPT_FIRST_CHUNK
                else:
                    previous_captions = captions[-1][1]
                    end = END_PROMPT_OTHER_CHUNKS.format(previous_captions=previous_captions)

                filtered_descriptions = replace_words(combined_descriptions, replacements)

                # Prepare the input for the local LLM
                input_text = prefix + filtered_descriptions + end

                print("INPUT TEXT")
                print(input_text)

                # Generate a creative caption
                output = llm(
                    f"system\n{input_text}\nassistant",  # The prompt format
                    max_tokens=4096,  # Generate up to 500 tokens
                    stop=["</s>"],  # Example stop token - not necessarily correct for this specific model! Please check before using.
                    echo=False,  # Do not echo the prompt
                )

                # Extract the generated caption
                caption = output['choices'][0]['text'] + ".."

                print(f"Creative Caption for chunk {chunk_idx}:")
                print(caption)
                captions.append((chunk_idx, caption))
 
            # Generate SRT file for the video
            subs = pysrt.SubRipFile()
            for chunk_idx, caption in captions:
                start_seconds = chunk_idx * interval_seconds
                end_seconds = start_seconds + interval_seconds
                subs.append(pysrt.SubRipItem(
                    index=chunk_idx+1,
                    start=pysrt.SubRipTime(seconds=start_seconds),
                    end=pysrt.SubRipTime(seconds=end_seconds),
                    text=caption
                ))

            srt_file_path = os.path.join('process', f'{video_file}.srt')
            subs.save(srt_file_path, encoding='utf-8')

            # Combine video and subtitles using ffmpeg
            output_video_path = os.path.join('cap-vid', f'final_output_{video_file}')
            ffmpeg_command = [
                'ffmpeg', '-i', video_path,
                '-vf', f"subtitles={srt_file_path}",
                output_video_path
            ]
            subprocess.run(ffmpeg_command, check=True)

            # Delete the original video
            os.remove(video_path)


        except Exception as e:
            print(f"An error occurred while processing {video_file}: {e}")
