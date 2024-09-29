import argparse
import torch
from video_captioning import VideoCaptioner
from commentary_generator import RAGSystem, AIAgent

if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description="Generate video commentary using VideoLLaVA+Gemma."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="example.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--VideoLLaVA_prompt",
        type=str,
        default="""
                USER: <video> Generate a concise live sports commentary that highlights events from the video such as key plays. Provide clear descriptions of the events. The commentary must strictly reflect the events in the video and avoid unnecessary details. Do not include speculative or fabricated details. ASSISTANT:
                """,
        help="Prompt for generating the video-captioning.",
    )
    parser.add_argument(
        "--Gemma_prompt",
        type=str,
        default="Make a script for a commentator's commentary on the {game} for {sport} {player} at the Paris Olympics based on video caption.",
        help="Prompt for generating the commentary.",
    )
    parser.add_argument(
        "--sport",
        type=str,
        default="diving men's 10m platform",
        help="Category of sports in the video, including specific details",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="final",
        help="Specific game or event within the sport (e.g., final, semifinal)",
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Chao Yuan",
        help="Name of the player or athlete featured in the video",
    )
    parser.add_argument(
        "--rag_data_type",
        type=str,
        default="json",
        help="Type of data used for Retrieval-Augmented Generation (RAG)",
    )
    parser.add_argument(
        "--num_retrieved_docs",
        type=int,
        default=3,
        help="Number of documents to retrieve for RAG",
    )

    args=parser.parse_args()

    # Video Captioning: Video-Llava
    VIDEO_CAP_MODEL_PATH = "LanguageBind/Video-LLaVA-7B-hf"

    video_captioner=VideoCaptioner(model_path=VIDEO_CAP_MODEL_PATH)
    VIDEO_CAPTION = video_captioner.generate_commentary(args.video_path, args.VideoLLaVA_prompt)

    # Free GPU memory after video captioning is complete
    torch.cuda.empty_cache()

    # Comentary Generator: Gemma
    COM_MODEL_PATH = "google/gemma-2-2b-it"

    if  args.rag_data_type == 'csv':
        RAG_PATH = "./data/sport_rag_data.csv"
        NUM_RETRIEVED_DOCS = 5
    else:
        RAG_PATH = "./data/sport_rag_data.json"
        NUM_RETRIEVED_DOCS = 3

    ai_agent = AIAgent(model_path=COM_MODEL_PATH)

    rag_system = RAGSystem(ai_agent, RAG_PATH, num_retrieved_docs=NUM_RETRIEVED_DOCS)

    gemma_prompt = args.Gemma_prompt.format(sport=args.sport, game=args.game, player=args.player, count=5)
    answer = rag_system.query(gemma_prompt, VIDEO_CAPTION)

    print(answer)
