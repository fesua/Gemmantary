import av
import numpy as np
import torch
import argparse
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from tqdm import tqdm

class VideoCaptioner:
    """
    Video-LLaVa video captioning
    It uses Video-LLaVa
    """
    def __init__(self, model_path):
        # Load the model and move it to CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto").to(self.device)
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)

    def read_video(self, video_path):
        """
        Function to read and sample 8 frames uniformly from a video.
        """
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, 8).astype(int)  # Sample 8 frames
        frames = [
            frame.to_ndarray(format="rgb24")
            for i, frame in enumerate(container.decode(video=0))
            if i in indices
        ]
        return np.stack(frames)

    def generate_commentary(self, video_path, prompt):
        # Step 1: Read the video and check if frames are loaded correctly
        video = self.read_video(video_path)
        if video is None or len(video) == 0:
            print("Error: Video frames were not properly loaded.")
            return "Error loading video frames."

        # Step 2: Process the inputs and check the processed input
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if inputs is None or len(inputs) == 0:
            print("Error: Inputs were not properly processed.")
            return "Error processing inputs."

        # Step 3: Generate output from the model using tqdm for progress
        print("Generating commentary...")
        with tqdm(total=500, desc="Generating output from the model") as pbar:
            output_ids = self.model.generate(**inputs, max_length=500)
            pbar.update(500)

        if output_ids is None or len(output_ids) == 0:
            print("Error: Model did not generate output.")
            return "Error generating output."

        # Step 4: Decode the output and check if the decoding is successful
        commentary = self.processor.batch_decode(output_ids, skip_special_tokens=True)

        if commentary is None or len(commentary) == 0:
            print("Error: Output could not be decoded.")
            return "Error decoding output."

        # Step 5: Extract the commentary part after "ASSISTANT:"
        if "ASSISTANT:" in commentary[0]:
            commentary = commentary[0].split("ASSISTANT:")[1].strip()
        else:
            commentary = commentary[0].strip()

        return commentary
