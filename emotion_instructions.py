"""
Emotion Instruction Library for Qwen3-TTS

Each emotion has multiple instruction variants to test.
We'll measure which instruction style produces the most consistent output.
"""

EMOTION_LIBRARY = {
    "neutral": {
        "description": "Default professional tone",
        "variants": [
            # Variant 0: Basic
            "Calm, even pace, professional neutrality, steady tone",

            # Variant 1: Physical descriptors
            "Speaking rate: 140 words per minute. Pitch: steady around 190Hz. Energy: consistent medium level. No emphasis on any words.",

            # Variant 2: Negative constraints
            "Calm and steady. Do NOT vary pitch between sentences. Do NOT add emphasis. Do NOT speed up or slow down. Maintain identical tone throughout.",

            # Variant 3: Reference style
            "Speak like reading a technical manual. Flat affect. Each sentence should sound identical in tone to the previous.",

            # Variant 4: Minimal
            "Monotone. Steady. Even.",
        ],
    },

    "warm": {
        "description": "Caring, empathetic tone",
        "variants": [
            "Warm, gentle, caring tone. Slight smile in voice. Soft energy.",
            "Speak with warmth and kindness. Slightly slower pace. Gentle, nurturing quality. Consistent soft energy throughout.",
            "Like comforting a friend. Warm but steady - do NOT vary intensity between sentences. Maintain the same caring tone throughout.",
            "Gentle warmth. Pitch slightly lower than neutral. Soft onset on each sentence. Consistent emotional tone.",
        ],
    },

    "excited": {
        "description": "Upbeat, enthusiastic",
        "variants": [
            "Excited, upbeat, enthusiastic. Higher energy, faster pace.",
            "Genuine excitement and enthusiasm. Speaking rate: 160 wpm. Pitch: elevated around 210Hz. Bright, energetic quality throughout.",
            "Upbeat and positive. Maintain HIGH energy consistently - do NOT drop energy between sentences. Each sentence equally enthusiastic.",
            "Like sharing great news. Bright, lifted tone. Fast but steady pace. Consistent excitement level.",
        ],
    },

    "concerned": {
        "description": "Gentle worry, care",
        "variants": [
            "Gentle concern, caring worry. Slower pace, soft tone.",
            "Speak with care and concern. Slightly slower: 120 wpm. Lower pitch around 175Hz. Soft, present quality.",
            "Concerned but calm. Show care without alarm. Steady, gentle delivery. Do NOT vary intensity - maintain consistent concerned tone.",
            "Like checking on someone's wellbeing. Soft, attentive, present. Even pacing throughout.",
        ],
    },

    "thoughtful": {
        "description": "Contemplative, considering",
        "variants": [
            "Thoughtful, contemplative, considering. Measured pace with slight pauses.",
            "Reflective and thoughtful. Speaking rate: 130 wpm. Natural pauses between phrases. Steady, contemplative quality.",
            "Like thinking out loud. Measured, deliberate. Do NOT rush. Maintain same thoughtful quality throughout all sentences.",
            "Contemplative. Slight pause before key words. Even, measured delivery. Consistent reflective tone.",
        ],
    },

    "reassuring": {
        "description": "Confident comfort",
        "variants": [
            "Reassuring, confident but gentle. Steady and calming.",
            "Speak with quiet confidence. Reassuring tone. Steady pace around 135 wpm. Calm, grounded energy throughout.",
            "Like calming someone's fears. Confident but soft. Do NOT waver - maintain steady reassuring tone. Each sentence equally grounding.",
            "Reassuring presence. Firm but gentle. Steady pitch, steady pace. Consistent calming quality.",
        ],
    },

    "professional": {
        "description": "Business-like, authoritative",
        "variants": [
            "Professional, business-like, clear and authoritative.",
            "Crisp, professional delivery. Speaking rate: 145 wpm. Clear enunciation. Confident, authoritative quality throughout.",
            "Like presenting to executives. Clear, direct, professional. Do NOT add emotional color. Maintain same business tone throughout.",
            "Corporate professional. Neutral affect. Clear articulation. Consistent authoritative tone.",
        ],
    },

    "empathetic": {
        "description": "Deep understanding",
        "variants": [
            "Deep empathy, understanding. Present, attentive, slower pace.",
            "Speak with genuine empathy. Slower: 115 wpm. Lower, softer pitch. Fully present, attentive quality.",
            "Like truly understanding someone's pain. Soft, present, unhurried. Do NOT rush. Maintain deep empathetic tone throughout.",
            "Empathetic presence. Soft, low energy. Gentle pace. Consistent understanding tone.",
        ],
    },

    "encouraging": {
        "description": "Supportive, uplifting",
        "variants": [
            "Encouraging, supportive, uplifting. Positive energy, warm.",
            "Speak with encouragement. Uplifting but not over-excited. Warm, supportive quality. Steady positive energy throughout.",
            "Like cheering someone on. Supportive, positive. Maintain consistent encouraging tone - do NOT vary between sentences.",
            "Encouraging coach. Warm, positive, steady. Consistent supportive energy.",
        ],
    },

    "playful": {
        "description": "Light, teasing",
        "variants": [
            "Playful, light, hint of teasing. Slightly faster, musical quality.",
            "Light and playful. Speaking rate: 150 wpm. Slight smile. Bouncy, musical intonation. Consistent playful quality.",
            "Like friendly banter. Light, teasing, fun. Do NOT get too serious. Maintain same playful energy throughout.",
            "Playful tease. Light energy. Slight lilt. Consistent fun tone.",
        ],
    },
}

# Test sentences that work for any emotion
NEUTRAL_TEST_CORPUS = [
    "The solar installation is scheduled for next Tuesday.",
    "I've reviewed the documents you sent over.",
    "Would you like me to check on that for you?",
    "That's not what I meant. Let me clarify.",
    "Let me walk you through the process step by step.",
]

# Emotion-appropriate test sentences
EMOTION_TEST_CORPUS = {
    "neutral": NEUTRAL_TEST_CORPUS,
    "warm": [
        "I really appreciate you sharing that with me.",
        "Thank you for trusting me with this.",
        "I'm here for you whenever you need.",
        "That means a lot, truly.",
        "I care about how this turns out for you.",
    ],
    "excited": [
        "Oh, that's fantastic news!",
        "I can't wait to see how this turns out!",
        "This is exactly what we were hoping for!",
        "What an amazing opportunity!",
        "I'm so thrilled about this!",
    ],
    "concerned": [
        "I want to make sure you're okay.",
        "That sounds like it might be difficult.",
        "Is everything alright?",
        "I'm a bit worried about that.",
        "Let me know if you need anything.",
    ],
    "thoughtful": [
        "Hmm, that's an interesting question.",
        "Let me think about that for a moment.",
        "There are several ways to look at this.",
        "I wonder if there's another approach.",
        "That's worth considering carefully.",
    ],
    "reassuring": [
        "Don't worry, we'll figure this out together.",
        "Everything is going to be okay.",
        "I've seen this before, and it's manageable.",
        "You're in good hands.",
        "We've got this under control.",
    ],
    "professional": [
        "Based on the data, here's what I found.",
        "The quarterly results show positive trends.",
        "I recommend we proceed with option two.",
        "Let me summarize the key points.",
        "The analysis indicates three main factors.",
    ],
    "empathetic": [
        "That sounds really difficult.",
        "I can only imagine how hard that must be.",
        "Your feelings are completely valid.",
        "I hear you, and I understand.",
        "That's a lot to carry.",
    ],
    "encouraging": [
        "You're doing great, keep going!",
        "I believe in you.",
        "You've got this!",
        "Look how far you've come already.",
        "Every step forward counts.",
    ],
    "playful": [
        "Oh, is that so?",
        "Well, well, well, look at you!",
        "Aren't you clever!",
        "I see what you did there.",
        "Now that's what I call interesting!",
    ],
}


def get_instruction(emotion: str, variant: int = 0) -> str:
    """Get instruction for an emotion with optional variant index."""
    if emotion not in EMOTION_LIBRARY:
        raise ValueError(f"Unknown emotion: {emotion}. Available: {list(EMOTION_LIBRARY.keys())}")

    variants = EMOTION_LIBRARY[emotion]["variants"]
    if variant >= len(variants):
        variant = 0

    return variants[variant]


def get_test_sentences(emotion: str) -> list[str]:
    """Get appropriate test sentences for an emotion."""
    return EMOTION_TEST_CORPUS.get(emotion, NEUTRAL_TEST_CORPUS)


def list_emotions() -> list[str]:
    """List all available emotions."""
    return list(EMOTION_LIBRARY.keys())


def list_variants(emotion: str) -> list[str]:
    """List all variants for an emotion."""
    if emotion not in EMOTION_LIBRARY:
        raise ValueError(f"Unknown emotion: {emotion}")
    return EMOTION_LIBRARY[emotion]["variants"]
