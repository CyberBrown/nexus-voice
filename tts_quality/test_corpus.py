"""Fixed test corpus for TTS quality evaluation."""

TEST_CORPUS = [
    # Neutral statements
    "The solar installation is scheduled for next Tuesday.",
    "I've reviewed the documents you sent over.",

    # Questions (rising intonation)
    "Would you like me to check on that for you?",
    "Have you considered the alternative approach?",

    # Emphasis/contrast
    "That's not what I meant. Let me clarify.",
    "The first option is faster, but the second is cheaper.",

    # Emotional content
    "I'm really excited about this opportunity!",
    "I understand this is frustrating. Let's work through it.",

    # Technical content
    "The API endpoint returns a 401 unauthorized error.",
    "Configure the inverter to output 240 volts AC.",

    # Multi-sentence (tests consistency across a response)
    "Let me walk you through the process. First, we'll need to verify your account. Then we can proceed with the installation scheduling.",
]

# Extended corpus for deeper testing
EXTENDED_CORPUS = [
    # More neutral statements
    "The meeting has been rescheduled to three o'clock.",
    "Your order will arrive within two business days.",
    "The system update completed successfully.",

    # Conversational
    "Oh, I see what you mean now.",
    "That makes a lot of sense, actually.",
    "Hmm, let me think about that for a moment.",

    # Informative
    "There are three main options to consider here.",
    "The primary difference is in the implementation approach.",
    "Based on the data, I'd recommend the second option.",

    # Empathetic
    "I completely understand your concern.",
    "That sounds like a challenging situation.",
    "I appreciate you sharing that with me.",

    # Technical explanations
    "The latency is measured in milliseconds from request to response.",
    "You'll need to authenticate using your API key.",
    "The model processes input tokens sequentially.",
]

# Sentences grouped by expected emotional tone
TONE_GROUPS = {
    "neutral": [
        "The solar installation is scheduled for next Tuesday.",
        "I've reviewed the documents you sent over.",
        "The meeting has been rescheduled to three o'clock.",
    ],
    "inquisitive": [
        "Would you like me to check on that for you?",
        "Have you considered the alternative approach?",
        "What would you like me to focus on first?",
    ],
    "emphatic": [
        "That's not what I meant. Let me clarify.",
        "The first option is faster, but the second is cheaper.",
        "This is really important to get right.",
    ],
    "warm": [
        "I'm really excited about this opportunity!",
        "I understand this is frustrating. Let's work through it.",
        "I appreciate you sharing that with me.",
    ],
    "technical": [
        "The API endpoint returns a 401 unauthorized error.",
        "Configure the inverter to output 240 volts AC.",
        "The latency is measured in milliseconds from request to response.",
    ],
}
