# Emotional-Support-Conversation-with-fine-grained-emotion-and-emotional-dynamics

## Problem Statement:
To improve the quality of responses generated in dialogue systems tailored for emotional support a machine learning model known as CauESC model needs to be developed, which enhances conversations in emotional support contexts by recognizing emotional causes and effects. By capturing the emotional dynamics between the seeker and support provider, CauESC offers a comprehensive solution for understanding and addressing emotional distress.

## Motivation/Faults in the previous similar models
Existing models for emotional support conversations face 2 major challenges:
- They ignore the emotional causes of distress.
- They focus solely on the seeker’s mental state without considering how emotions shift during interactions.
The CauESC model tackles these issues, capturing both emotional causes and effects, resulting in improved response generation.

## Objectives
The main objectives of this project are to:
1. Develop the CauESC model that analyzes both the causes and effects of emotions in support conversations.
2. Incorporate an attention mechanism to simulate human-like reasoning about emotional dynamics.
3. Propose a "multi-strategy" approach for emotional support dialogue generation.
4. Achieve state-of-the-art performance in strategy selection and response generation using the *ESConv dataset*.

## Methodology
The *CauESC* model consists of three main components:
1. Cause Aware Encoder: Detects emotional causes and captures the emotional dynamics.
2. Causal Interaction Module: Provides fine-grained emotion understanding by reasoning through cause-effect relationships.
3. Independent-Integrated Strategy Executors: Balances independent and integrated strategies to guide response generation.

## Key Features
- Emotion Cause Detector: Focuses on identifying the causes of distress.
- COMET Integration: Uses **COMET (Bosselut et al., 2019) to understand the effects of emotional causes.
- Novel Decoder: Balances multiple strategies for comprehensive emotional support.

## References
- Bosselut et al., 2019. [COMET: The Commonsense Transformers](https://arxiv.org/pdf/2401.17755).
- [GitHub Repository](https://github.com/Akshatjamadagni/Emotional-Support-Conversation-with-fine-grained-emotion-and-emotional-dynamics).

License: This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
