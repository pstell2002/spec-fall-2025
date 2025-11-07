def build_prompt(problem: str) -> str:
    """
    Requested prompt:

    ###
    Question: _____

    ###
    Answer: “Let’s think step by step...
    """
    return f"""###
Question: {problem}

###
Answer: “Let’s think step by step..."""
