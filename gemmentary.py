import argparse
import torch
from video_captioning import VideoCaptioner
from commentary_generator import RAGSystem, AIAgent

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description="Generate video commentary using VideoLLaVA+Gemma."
    )
    parser.add_argument("--video_path", type=str, help="example.mp4")
    parser.add_argument(
        "--VideoLLaVA_prompt",
        type=str,
        default="""
                USER: <video> Generate a concise live sports commentary that highlights events from the video such as key plays. Provide clear descriptions of the events. The commentary must strictly reflect the events in the video and avoid unnecessary details. Do not include speculative or fabricated details. ASSISTANT:
                """,
        help="Prompt for generating the commentary.",
    )
    parser.add_argument('--sport', type=str, default="table tennis women's singles")
    parser.add_argument('--game', type=str, default="final")
    parser.add_argument('--player', type=str, default="Chao Yuan")
    parser.add_argument('--rag_data_type', type=str, default='json')
    parser.add_argument('--num_retrieved_docs', type=int, default=3)

    args=parser.parse_args()

    # Video Captioning: Video-Llava

    video_captioner=VideoCaptioner()
    VIDEO_CAPTION = video_captioner.generate_commentary(args.video_path, args.prompt)

    # Comentary Generator: Gemma

    MODEL_PATH = "google/gemma-2b-it"

    if  args.rag_data_type == 'csv':
        RAG_PATH = "./data/sport_rag_data.csv"
        NUM_RETRIEVED_DOCS = 5
    else:
        RAG_PATH = "./data/sport_rag_data.json"
        NUM_RETRIEVED_DOCS = 3
    
    ai_agent = AIAgent(model_path=MODEL_PATH)

    rag_system = RAGSystem(ai_agent, RAG_PATH, num_retrieved_docs=NUM_RETRIEVED_DOCS)

    prompt = "Make a script for a commentator's commentary on the {game} for {sport} {player} at the Paris Olympics based on video caption."
    prompt = prompt.format(sport=args.sport, game=args.game, player=args.player, count=5)
    answer = rag_system.query(prompt, VIDEO_CAPTION)

    