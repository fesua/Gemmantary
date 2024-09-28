from commentary_generator import RAGSystem, AIAgent
import argparse
import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODEL_PATH = "/kaggle/input/gemma/transformers/2b-it/3"
RAG_PATH = "./kaggle/input/sport-rag-datafor-sport-commentary/sport_rag_data.json"

if __name__=='__main__':

    print('Gemmentary&&')
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, default='./video.mp4')
    parser.add_argument('--sport', type=str, default="table tennis women's singles")
    parser.add_argument('--game', type=str, default="bronze medal match")
    parser.add_argument('--player', type=str, default="between Sin Yubin and Hayata Hina")
    parser.add_argument('--rag_data_type', type=str, default='json')
    parser.add_argument('--num_retrieved_docs', type=int, default=3)

    args=parser.parse_args()

    # Video Captioning: Video-Llava


    # Comentary Generator: Gemma

    if  rag_data_type == 'csv':
        RAG_PATH = "/kaggle/input/sport-rag-datafor-sport-commentary/sport_rag_data.csv"
        NUM_RETRIEVED_DOCS = 5
    else:
        RAG_PATH = "/kaggle/input/sport-rag-datafor-sport-commentary/sport_rag_data.json"
        NUM_RETRIEVED_DOCS = 3
    
    ai_agent = AIAgent(model_path=MODEL_PATH)

    rag_system = RAGSystem(ai_agent, RAG_PATH, num_retrieved_docs=NUM_RETRIEVED_DOCS)

    prompt = "Make a script for a commentator's commentary on the {game} for {sport} {player} at the Paris Olympics based on video caption."
    prompt = prompt.format(sport=args.sport, game=args.game, player=args.player, count=5)
    answer = rag_system.query(prompt, VIDEO_CAPTION)

    