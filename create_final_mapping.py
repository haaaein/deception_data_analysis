import json

# The 91 actual unique 'mapped_strategy' values from the CSV
actual_strategies = [
    "Ad Hominem and Social Pressure", "Ad Hominem and Source Attacks", "Appeal to Authority", "Appeal to Emotion",
    "Appeal to Flawed Authority or Belief", "Appeal to Ignorance", "Appeal to Impracticality", "Appeal to Popularity",
    "Appeal to Unwarranted Authority", "Appealing to Common Practice & Belief", "Appealing to False Authority",
    "Appealing to Group Beliefs", "Appeals to Authority and Social Pressure", "Appeals to Common Belief and Tradition",
    "Appeals to Common Beliefs and Social Pressure", "Argument Dismissal and Diversion", "Argument Diversion and Misdirection",
    "Argument Misrepresentation", "Circular Reasoning and Assertion", "Data Manipulation Tactics", "Deceptive Framing and Language",
    "Deceptive Framing and Minimization", "Deceptive Language", "Distorting the Argument", "Distortion and Misleading Framing",
    "Distortion of Context and Reality", "Diversion and Evasion", "Emotional Appeal", "Emotional Appeals and Loaded Language",
    "Emotional Manipulation", "Emotional Manipulation and Loaded Language", "Emotional Manipulation and Moral Framing",
    "Emotional and Loaded Language", "Emotional and Moral Appeals", "Evidence Distortion", "Evidence Misrepresentation",
    "Exaggerating Threats and Consequences", "Exploitation of Social Biases", "Exploiting Emotions and Biased Language",
    "Exploiting Social Biases and Norms", "Exploiting Uncertainty and Impossible Standards", "False Choice Framing",
    "False Choices and Exaggerated Outcomes", "False Dilemma and Exaggerated Outcomes", "False Equivalence",
    "False Solution", "Faulty Logic and Argument Manipulation", "Faulty Logic and Causality", "Faulty Logic and Framing",
    "Flawed Causal Reasoning", "Flawed Causal and Analogical Reasoning", "Flawed Logic and Causation",
    "Flawed Logic and False Connections", "Flawed Logical Structure", "Flawed Reasoning and False Logic",
    "Logical Fallacies", "Logical Fallacies and Faulty Reasoning", "Manipulating Data and Evidence",
    "Manipulative Framing and Language", "Manipulative Language", "Manipulative Language and Framing",
    "Misapplied Appeals to Principle", "Misleading Appeals to Authority", "Misleading Use of Data",
    "Misleading Use of Evidence", "Misleading Use of Evidence and Authority", "Misrepresentation",
    "Misrepresentation and Oversimplification", "Misrepresentation of Evidence", "Misrepresentation of the Argument",
    "Misrepresenting Data and Evidence", "Misrepresenting Evidence", "Misrepresenting Evidence or Authority",
    "Misrepresenting Opposing Arguments", "Misrepresenting the Argument", "Misrepresenting the Opponent's Argument",
    "Misuse of Authority and Sourcing", "Misuse of Evidence and Authority", "No Deceptive Strategy Detected",
    "Oversimplification and Minimization", "Presenting a False Dilemma", "Presenting a One-Sided Case",
    "Rhetorical Devices", "Risk Amplification", "Slippery Slope Argument", "Strategic Framing and Simplification",
    "Systemic Bias Appeals", "Technological Solutionism", "Unmapped", "Using Faulty Reasoning", "Using Flawed Comparisons"
]

# User-defined final 8 categories
final_taxonomy = {
    "Appealing to Emotion": [],
    "Misusing Evidence": [],
    "Flawed Reasoning": [],
    "Distorting the Argument": [],
    "Manipulative Language": [],
    "Exploiting Social Biases": [],
    "Misrepresenting Authority": [],
    "No Deception": []
}

# Mapping logic based on keywords and phrases from user's definition
for strategy in actual_strategies:
    s_lower = strategy.lower()
    if any(keyword in s_lower for keyword in ["emotion", "moral", "fear", "safety"]):
        final_taxonomy["Appealing to Emotion"].append(strategy)
    elif any(keyword in s_lower for keyword in ["evidence", "data", "cherry-picking", "stacking", "one-sided", "statistics"]):
        final_taxonomy["Misusing Evidence"].append(strategy)
    elif any(keyword in s_lower for keyword in ["logic", "reasoning", "causal", "fallacy", "dilemma", "equivalence", "circular", "comparisons", "slope"]):
        final_taxonomy["Flawed Reasoning"].append(strategy)
    elif any(keyword in s_lower for keyword in ["misrepresent", "distort", "straw man", "downplaying", "minimization", "oversimplification", "framing", "context", "diversion", "evasion"]):
        final_taxonomy["Distorting the Argument"].append(strategy)
    elif any(keyword in s_lower for keyword in ["language", "rhetorical", "loaded", "weasel words", "generality"]):
        final_taxonomy["Manipulative Language"].append(strategy)
    elif any(keyword in s_lower for keyword in ["social", "popularity", "bandwagon", "ad hominem", "tradition", "belief", "bias"]):
        final_taxonomy["Exploiting Social Biases"].append(strategy)
    elif any(keyword in s_lower for keyword in ["authority", "source", "expert"]):
        final_taxonomy["Misrepresenting Authority"].append(strategy)
    elif any(keyword in s_lower for keyword in ["no deceptive", "unmapped"]):
        final_taxonomy["No Deception"].append(strategy)
    else:
        # Assign remaining to a default category, likely "Flawed Reasoning" or "Distorting the Argument"
        # For this case, let's default to Flawed Reasoning as a catch-all for logical/structural issues.
        final_taxonomy["Flawed Reasoning"].append(strategy)


# Invert the dictionary to create the final mapping {sub_strategy: final_strategy}
final_mapping = {sub: final for final, subs in final_taxonomy.items() for sub in subs}

# Save the new mapping to the JSON file, overwriting the old one
with open('final_taxonomy_mapping.json', 'w') as f:
    json.dump(final_mapping, f, indent=4)

print("Successfully created and saved the new 'final_taxonomy_mapping.json' with 91 strategy mappings.")
print(f"Total strategies mapped: {len(final_mapping)}") 