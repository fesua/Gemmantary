# Gemmantary-sports commentary with gemma

We conducted a project that takes Olympic videos as input and generates commentary based on them.


<div align="center">
  <a target="_blank" href="https://colab.research.google.com/drive/1PuezmbUfrJPvqJUM5qnOuUm5kPrbUV6X?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</div>

<br>

## Example

\<User\>: Make a script for a commentator's commentary on the final for the diving men's 10m platform for diving Chao Yuan at the Paris Olympics based on video caption.


<br>

https://github.com/user-attachments/assets/49c77908-a89a-42df-8062-8d231df58526


<br>

Good evening, sports fans, and welcome to the final of the diving men's 10m platform at the Paris Olympics. We have a battle for gold between two of the greatest gymnasts of all time, Chao Yuan from China and the defending champion, [Name of the competitor from another country].

The Dive "The athletes take their positions on the platform, and the crowd is on its feet as the music starts. Chao Yuan jumps into the air with incredible precision and grace, his body arching as he prepares to make the dive. He executes a perfect somersault dive, showcasing his incredible athleticism and control. The crowd is on its feet, and the judges are watching the performance with rapt attention."

The Aftermath "The impact is visually stunning as Chao Yuan hits the water with a splash, creating a moment of pure exhilaration. His body is perfectly arched, and his performance is a testament to the beauty of the sport. The judges give him a standing ovation for his incredible performance."

Conclusion "And that concludes the final of the diving men's 10m platform at the Paris Olympics. Chao Yuan has secured his gold medal, and the crowd erupts in applause. It's been an incredible final, and we've witnessed some of the best diving the world has ever seen. Thank you for joining us for the coverage of the Paris Olympics."


<br>

## We use 'Video-LLaVA' and 'LangChain' and 'Gemma'

Video-LLaVA: https://github.com/PKU-YuanGroup/Video-LLaVA 

LangChain: https://www.langchain.com/

Gemma: https://huggingface.co/google/gemma-2-2b-it

<br>

## Our Pipeline

Our pipeline is as follows:

1. Input: The user provides a video, the sport type, the name of the athlete, and the event information.

2. Formatting: We place the sport type, athlete name, and event information into a custom format we designed.

3. Video Description: Using Video-LLaVA, we generate a description of the video content.

4. Commentary Generation: We integrate the video description with our custom format through RAG (Retrieval-Augmented Generation) and generate commentary using the Gemma model.


<br>

## How to run our code
