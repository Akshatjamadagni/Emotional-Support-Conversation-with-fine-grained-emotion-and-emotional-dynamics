# ESConv Dataset

The **ESConv.json** dataset is designed to support research in emotionally supportive dialogue systems. It provides annotated conversations where support seekers share their emotional experiences, and supporters respond using strategies to provide emotional support.

---

## Dataset Overview

This dataset features:
- **Emotion Types**: Includes diverse emotional challenges such as anxiety, anger, sadness, and more.
- **Support Strategies**: Conversations are annotated with strategies such as *Questioning*, *Affirmation and Reassurance*, *Self-Disclosure*, *Providing Suggestions*, and others.
- **Structured Dialogues**: Each entry is organized into a dialogue format with speaker roles (seeker and supporter) and corresponding annotations.
- **Survey Scores**: Feedback from both seekers and supporters to assess empathy, relevance, and emotional intensity shifts.

---

## File Structure

The dataset is provided in **JSON** format, with a sample structure as follows:

### Format:
```json
[
  {
    "experience_type": "Previous Experience",
    "emotion_type": "anxiety",
    "problem_type": "job crisis",
    "situation": "I hate my job but I am scared to quit and seek a new career.",
    "survey_score": {
      "seeker": {
        "initial_emotion_intensity": "5",
        "empathy": "5",
        "relevance": "5",
        "final_emotion_intensity": "1"
      },
      "supporter": {
        "relevance": "5"
      }
    },
    "dialog": [
      {
        "speaker": "seeker",
        "annotation": {},
        "content": "Hello"
      },
      {
        "speaker": "supporter",
        "annotation": {
          "strategy": "Question"
        },
        "content": "Hello, what would you like to talk about?"
      }
    ],
    "seeker_question1": "Partner was very supportive",
    "seeker_question2": "More guidance in conversation or examples",
    "supporter_question1": "",
    "supporter_question2": ""
  }
]
```

### Key Fields:
1. **experience_type**:
   - Denotes whether the experience is current or past.
   - Example: `"Previous Experience"`, `"Current Experience"`

2. **emotion_type**:
   - The emotional state of the seeker.
   - Example: `"anxiety"`, `"anger"`

3. **problem_type**:
   - The nature of the seeker's problem.
   - Example: `"job crisis"`, `"problems with friends"`

4. **situation**:
   - A brief description of the seeker's problem.
   - Example: `"I hate my job but I am scared to quit and seek a new career."`

5. **survey_score**:
   - Feedback scores from the seeker and supporter:
     - **initial_emotion_intensity**: The seeker's emotional intensity at the start of the conversation.
     - **final_emotion_intensity**: The seeker's emotional intensity at the end.
     - **empathy** and **relevance**: Ratings for the supporter’s response.

6. **dialog**:
   - The dialogue exchanged between the seeker and the supporter.
   - **speaker**: Indicates whether the message is from the seeker or supporter.
   - **annotation**: Includes the strategy employed by the supporter (if any).
   - **content**: The text of the message.

7. **Seeker/Supporter Questions**:
   - Questions capturing feedback or additional context.

---

## Usage

### Loading the Dataset
You can load and process the dataset using Python:

```python
import json

# Load ESConv dataset
with open("ESConv.json", "r",encoding='utf-8') as file:
    data = json.load(file)

# Example: Print the first conversation
print(data[0])
```

### Potential Applications
- **Emotionally Supportive AI**:
  - Fine-tune models for empathetic responses.
  - Train conversational agents for mental health support.
- **Psychology and Counseling**:
  - Analyze how specific support strategies affect emotional outcomes.
- **Human-Computer Interaction**:
  - Build AI systems with enhanced emotional intelligence.

---

## Annotation Strategies

The dataset annotates supporter messages with one or more strategies, including:  

1. **Question (Qu):** Asking clarifying, probing, or open-ended questions to encourage the seeker to share more or explore their thoughts.  
2. **Restatement or Paraphrasing (RP):** Reiterating or rephrasing the seeker’s statements to ensure understanding or validate their perspective.  
3. **Reflection of Feelings (RF):** Identifying and acknowledging the emotions expressed by the seeker to show empathy and understanding.  
4. **Self-Disclosure (Sd):** Sharing personal experiences to create a connection or provide relatable insights.  
5. **Affirmation and Reassurance (AR):** Validating the seeker’s feelings or providing comforting and reassuring statements.  
6. **Providing Suggestions (PS):** Offering actionable advice, potential solutions, or recommendations to address the seeker’s concerns.  
7. **Information (In):** Sharing factual, educational, or informative content to aid the seeker in understanding their situation or options.  
8. **Others (Ot):** Any other strategy not explicitly categorized, including greetings, pleasantries, or context-specific interactions.

---
