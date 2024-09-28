# video-llava.py

import av
import numpy as np
import torch
import argparse
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download

# Load the model and move it to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto"
)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")


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


def generate_commentary(video_path, prompt):
    video = read_video(video_path)
    inputs = processor(text=prompt, videos=video, return_tensors="pt")

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output from the model
    output_ids = model.generate(**inputs, max_length=300)

    # Decode the output to human-readable text
    commentary = processor.batch_decode(output_ids, skip_special_tokens=True)

    # Get the text after "ASSISTANT:" only
    if "ASSISTANT:" in commentary[0]:
        commentary = commentary[0].split("ASSISTANT:")[1].strip()
    else:
        commentary = commentary[0].strip()

    return commentary


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate video commentary using VideoLLaVA."
    )
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="""
    USER: <video> Generate a concise live sports commentary that highlights events from the video such as key plays. Provide clear descriptions of the events. The commentary must strictly reflect the events in the video and avoid unnecessary details. Do not include speculative or fabricated details. ASSISTANT:
    """,
        help="Prompt for generating the commentary.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the commentary generation
    commentary = generate_commentary(args.video_path, args.prompt)

    # Output the commentary
    print(commentary)
