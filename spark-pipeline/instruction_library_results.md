# Instruction Library Test Results

**Date:** 2026-01-28 23:14:28
**Model:** Fine-tuned Samantha (via TTS server on port 8032)
**Target:** Spectral similarity > 0.85
**Emotions Tested:** 10
**Total Variants:** 41

## Summary: Best Instruction Per Emotion

| Emotion | Best Variant | Spectral Sim | Pitch Var | Energy Var |
|---------|--------------|--------------|-----------|------------|
| neutral | 3 | 0.850 | 0.00 Hz | 0.0032 |
| warm | 0 | 0.853 | 0.00 Hz | 0.0034 |
| excited | 3 | 0.873 | 0.00 Hz | 0.0041 |
| concerned | 1 | 0.876 | 0.00 Hz | 0.0036 |
| thoughtful | 1 | 0.884 | 0.00 Hz | 0.0051 |
| reassuring | 2 | 0.754 | 0.00 Hz | 0.0035 |
| professional | 3 | 0.819 | 0.00 Hz | 0.0036 |
| empathetic | 1 | 0.783 | 0.00 Hz | 0.0029 |
| encouraging | 1 | 0.776 | 0.00 Hz | 0.0023 |
| playful | 1 | 0.798 | 0.00 Hz | 0.0045 |

**Overall Best:** thoughtful variant 1 at 0.884

## HIT TARGET (>0.85)
- thoughtful variant 1: 0.884
- concerned variant 1: 0.876
- excited variant 3: 0.873
- excited variant 2: 0.868
- concerned variant 2: 0.866
- thoughtful variant 0: 0.859
- excited variant 0: 0.856
- thoughtful variant 3: 0.855
- warm variant 0: 0.853
- neutral variant 3: 0.850

## Detailed Results by Emotion

### Neutral

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 3 | Speak like reading a technical manual. F... | 0.850 | 0.00 Hz |
| 4 | Monotone. Steady. Even.... | 0.826 | 0.00 Hz |
| 1 | Speaking rate: 140 words per minute. Pit... | 0.826 | 0.00 Hz |
| 0 | Calm, even pace, professional neutrality... | 0.818 | 0.00 Hz |
| 2 | Calm and steady. Do NOT vary pitch betwe... | 0.815 | 0.00 Hz |

### Warm

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 0 | Warm, gentle, caring tone. Slight smile ... | 0.853 | 0.00 Hz |
| 3 | Gentle warmth. Pitch slightly lower than... | 0.828 | 0.00 Hz |
| 1 | Speak with warmth and kindness. Slightly... | 0.815 | 0.00 Hz |
| 2 | Like comforting a friend. Warm but stead... | 0.801 | 0.00 Hz |

### Excited

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 3 | Like sharing great news. Bright, lifted ... | 0.873 | 0.00 Hz |
| 2 | Upbeat and positive. Maintain HIGH energ... | 0.868 | 0.00 Hz |
| 0 | Excited, upbeat, enthusiastic. Higher en... | 0.856 | 0.00 Hz |
| 1 | Genuine excitement and enthusiasm. Speak... | 0.830 | 0.00 Hz |

### Concerned

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 1 | Speak with care and concern. Slightly sl... | 0.876 | 0.00 Hz |
| 2 | Concerned but calm. Show care without al... | 0.866 | 0.00 Hz |
| 3 | Like checking on someone's wellbeing. So... | 0.845 | 0.00 Hz |
| 0 | Gentle concern, caring worry. Slower pac... | 0.817 | 0.00 Hz |

### Thoughtful

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 1 | Reflective and thoughtful. Speaking rate... | 0.884 | 0.00 Hz |
| 0 | Thoughtful, contemplative, considering. ... | 0.859 | 0.00 Hz |
| 3 | Contemplative. Slight pause before key w... | 0.855 | 0.00 Hz |
| 2 | Like thinking out loud. Measured, delibe... | 0.829 | 0.00 Hz |

### Reassuring

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 2 | Like calming someone's fears. Confident ... | 0.754 | 0.00 Hz |
| 3 | Reassuring presence. Firm but gentle. St... | 0.736 | 0.00 Hz |
| 0 | Reassuring, confident but gentle. Steady... | 0.711 | 0.00 Hz |
| 1 | Speak with quiet confidence. Reassuring ... | 0.711 | 0.00 Hz |

### Professional

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 3 | Corporate professional. Neutral affect. ... | 0.819 | 0.00 Hz |
| 2 | Like presenting to executives. Clear, di... | 0.802 | 0.00 Hz |
| 0 | Professional, business-like, clear and a... | 0.785 | 0.00 Hz |
| 1 | Crisp, professional delivery. Speaking r... | 0.770 | 0.00 Hz |

### Empathetic

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 1 | Speak with genuine empathy. Slower: 115 ... | 0.783 | 0.00 Hz |
| 3 | Empathetic presence. Soft, low energy. G... | 0.763 | 0.00 Hz |
| 0 | Deep empathy, understanding. Present, at... | 0.758 | 0.00 Hz |
| 2 | Like truly understanding someone's pain.... | 0.729 | 0.00 Hz |

### Encouraging

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 1 | Speak with encouragement. Uplifting but ... | 0.776 | 0.00 Hz |
| 0 | Encouraging, supportive, uplifting. Posi... | 0.772 | 0.00 Hz |
| 3 | Encouraging coach. Warm, positive, stead... | 0.767 | 0.00 Hz |
| 2 | Like cheering someone on. Supportive, po... | 0.758 | 0.00 Hz |

### Playful

| Variant | Instruction (truncated) | Spectral Sim | Pitch Var |
|---------|------------------------|--------------|------------|
| 1 | Light and playful. Speaking rate: 150 wp... | 0.798 | 0.00 Hz |
| 0 | Playful, light, hint of teasing. Slightl... | 0.764 | 0.00 Hz |
| 3 | Playful tease. Light energy. Slight lilt... | 0.710 | 0.00 Hz |
| 2 | Like friendly banter. Light, teasing, fu... | 0.709 | 0.00 Hz |

## Recommended Instruction Library

```python
BEST_INSTRUCTIONS = {
    "neutral": "Speak like reading a technical manual. Flat affect. Each sentence should sound identical in tone to the previous.",
    "warm": "Warm, gentle, caring tone. Slight smile in voice. Soft energy.",
    "excited": "Like sharing great news. Bright, lifted tone. Fast but steady pace. Consistent excitement level.",
    "concerned": "Speak with care and concern. Slightly slower: 120 wpm. Lower pitch around 175Hz. Soft, present quality.",
    "thoughtful": "Reflective and thoughtful. Speaking rate: 130 wpm. Natural pauses between phrases. Steady, contemplative quality.",
    "reassuring": "Like calming someone's fears. Confident but soft. Do NOT waver - maintain steady reassuring tone. Each sentence equally grounding.",
    "professional": "Corporate professional. Neutral affect. Clear articulation. Consistent authoritative tone.",
    "empathetic": "Speak with genuine empathy. Slower: 115 wpm. Lower, softer pitch. Fully present, attentive quality.",
    "encouraging": "Speak with encouragement. Uplifting but not over-excited. Warm, supportive quality. Steady positive energy throughout.",
    "playful": "Light and playful. Speaking rate: 150 wpm. Slight smile. Bouncy, musical intonation. Consistent playful quality.",
}
```

## Instruction Pattern Analysis

Analyzing which instruction styles perform best:

| Style | Avg Spectral Sim | Min | Max | Count |
|-------|------------------|-----|-----|-------|
| basic | 0.799 | 0.711 | 0.859 | 10 |
| physical_descriptors | 0.807 | 0.711 | 0.884 | 10 |
| negative_constraints | 0.793 | 0.709 | 0.868 | 10 |
| reference_style | 0.805 | 0.710 | 0.873 | 10 |
| minimal | 0.826 | 0.826 | 0.826 | 1 |

**Best instruction style overall:** minimal (avg: 0.826)

## Key Insights

### Instruction Style Ranking (by avg spectral similarity)
1. **minimal**: 0.826
2. **physical_descriptors**: 0.807
3. **reference_style**: 0.805
4. **basic**: 0.799
5. **negative_constraints**: 0.793

### Low Pitch Variance (<15 Hz): 41 variants
Best with low pitch variance: thoughtful v1 (0.884)
