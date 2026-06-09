"""
Few-shot pipeline: include K examples per tier in every prompt so the model
has concrete anchors, then predict price_tier and evaluate Macro F1-Score.
"""

import re
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FILE  = "data/train_01.csv"
TEST_FILE   = "data/test_01.csv"
MODEL_NAME  = "llama3.2"
VAL_SIZE    = 0.1   # fraction of train used for local F1 eval
SAMPLE_N    = None  # set to e.g. 50 to run on a small subset first
SHOTS_PER_TIER = 3  # examples per tier injected into every prompt (3×4 = 12 total)

TIER_LABELS = {0: "Budget", 1: "Standard", 2: "Premium", 3: "Ultra-Luxury"}


def build_few_shot_block(train_df: pd.DataFrame) -> str:
    """Pick SHOTS_PER_TIER examples per tier, then shuffle to avoid ordering bias."""
    rows = []
    for tier in range(4):
        sample = (
            train_df[train_df["price_tier"] == tier]
            .sample(SHOTS_PER_TIER, random_state=42)
        )
        for _, r in sample.iterrows():
            rows.append((
                f"  Description: {r['description']} | "
                f"Neighbourhood group: {r['neighbourhood_group']} | "
                f"Neighbourhood: {r['neighbourhood']} | "
                f"Room type: {r['room_type']} | "
                f"Min nights: {r['minimum_nights']} | "
                f"Reviews: {r['number_of_reviews']} | "
                f"Availability: {r['availability_365']} "
                f"→ {tier}"
            ))
    # shuffle so the model doesn't anchor on the last-seen tier
    import random
    random.seed(42)
    random.shuffle(rows)
    return "\n".join(rows)


# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = PromptTemplate(
    input_variables=["few_shot_examples", "description", "neighbourhood_group",
                     "neighbourhood", "room_type", "minimum_nights",
                     "number_of_reviews", "availability_365"],
    template="""You are a New York City real estate expert classifying Airbnb listings into price tiers.

Price tiers and their typical signals:
  0 = Budget       — shared or small private room, outer boroughs (Bronx/Queens/Staten Island), \
high minimum nights, very few reviews or very high availability, words like "cozy closet", "shared bath"
  1 = Standard     — entire apartment or private room, mixed neighbourhoods, moderate reviews, \
average availability, no luxury signals
  2 = Premium      — desirable Manhattan/Brooklyn neighbourhoods (SoHo, Williamsburg, Chelsea), \
entire home, low minimum nights, words like "stunning", "renovated", "skyline view"
  3 = Ultra-Luxury — penthouse, loft, townhouse, top Manhattan neighbourhoods (Upper West Side, \
Tribeca, West Village), near-zero availability (always booked), words like "marble", "doorman", \
"rooftop", "luxury"

Note: tiers 0 and 1 are equally common; tier 2 is slightly more common; tier 3 is rare (~10%). \
Do NOT default to tier 2 — read the signals carefully.

Labeled examples (shuffled):
{few_shot_examples}

Now classify this listing:
  Description        : {description}
  Neighbourhood group: {neighbourhood_group}
  Neighbourhood      : {neighbourhood}
  Room type          : {room_type}
  Minimum nights     : {minimum_nights}
  Number of reviews  : {number_of_reviews}
  Availability/365   : {availability_365}

Reply with ONLY a single digit: 0, 1, 2, or 3. No explanation."""
)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = OllamaLLM(model=MODEL_NAME, temperature=0)
chain = PROMPT | llm


def parse_tier(response: str) -> int:
    match = re.search(r"[0-3]", response.strip())
    return int(match.group()) if match else 1


def predict_rows(df: pd.DataFrame, few_shot_block: str) -> list[int]:
    predictions = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        response = chain.invoke({
            "few_shot_examples":   few_shot_block,
            "description":         str(row.description),
            "neighbourhood_group": str(row.neighbourhood_group),
            "neighbourhood":       str(row.neighbourhood),
            "room_type":           str(row.room_type),
            "minimum_nights":      str(row.minimum_nights),
            "number_of_reviews":   str(row.number_of_reviews),
            "availability_365":    str(row.availability_365),
        })
        pred = parse_tier(response)
        predictions.append(pred)
        if i % 10 == 0:
            print(f"  [{i}/{len(df)}] last response: {response.strip()!r} → {pred}")
    return predictions


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    # Local validation split — few-shot examples are drawn only from train_split
    train_split, val_split = train_test_split(
        train_df, test_size=VAL_SIZE, random_state=42, stratify=train_df["price_tier"]
    )

    few_shot_block = build_few_shot_block(train_split)

    if SAMPLE_N:
        val_split = val_split.sample(SAMPLE_N, random_state=42)

    print(f"Validation set: {len(val_split)} rows  |  Few-shot examples: {SHOTS_PER_TIER * 4}")
    print("Running predictions …\n")

    preds = predict_rows(val_split.reset_index(drop=True), few_shot_block)
    true  = val_split["price_tier"].tolist()

    f1 = f1_score(true, preds, average="macro")
    print(f"\nMacro F1-Score (validation): {f1:.4f}")

    # Predict on test set using all train data for few-shot examples
    full_few_shot_block = build_few_shot_block(train_df)
    print(f"\nPredicting test set ({len(test_df)} rows) …")
    test_preds = predict_rows(test_df.reset_index(drop=True), full_few_shot_block)
    test_df["price_tier"] = test_preds
    output_file = "outputs/predictions.csv"
    test_df[["property_id", "price_tier"]].to_csv(output_file, index=False)
    print(f"Saved → {output_file}")


if __name__ == "__main__":
    main()
