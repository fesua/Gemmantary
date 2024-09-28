import av
import numpy as np
import torch
import argparse
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download

class VideoCaptioner:
    """
    Video-LLaVa video captioning
    It uses Video-LLaVa
    """
    def __init__(self):
        # Load the model and move it to CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            "LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto"
            )
        self.processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    def read_video(video_path):
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
        video = self.read_video(video_path)
        inputs = self.processor(text=prompt, videos=video, return_tensors="pt")

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate output from the model
        output_ids = self.model.generate(**inputs, max_length=300)

        # Decode the output to human-readable text
        commentary = self.processor.batch_decode(output_ids, skip_special_tokens=True)

        # Get the text after "ASSISTANT:" only
        if "ASSISTANT:" in commentary[0]:
            commentary = commentary[0].split("ASSISTANT:")[1].strip()
        else:
            commentary = commentary[0].strip()

        return commentary