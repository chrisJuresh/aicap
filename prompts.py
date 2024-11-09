STORY = "Add a sentence or two describing what the video is about"

END_PROMPT_FIRST_CHUNK = (
    "\n The list above is a description of each frame of 20 seconds of a video guessed by an AI. "
    f"\n{STORY} \nWrite a short caption directed at me using basic english to describe what is happening in the video."
)

END_PROMPT_OTHER_CHUNKS = (
    "\n The list above is a description of each frame of 20 seconds of a video guessed by an AI. "
    f"\n{STORY} \nWrite a short caption directed at me using basic english to describe what is happening in the video. "
    "\n The caption must flow as a story, based on these previous captions: {previous_captions} "
    "\nContinue the next sentence in the story:"
)

# Dictionary
REPLACEMENTS = {
    "word_to_remove": "word_to_substitute_in",
    "example2": "replacement2"
}