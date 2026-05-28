import requests
import csv
import re
import time


BASE_URL       = "http://localhost:1234"
CHAT_URL       = f"{BASE_URL}/v1/chat/completions"
LOAD_URL       = f"{BASE_URL}/api/v1/models/load"
UNLOAD_URL     = f"{BASE_URL}/api/v1/models/unload"

REPETITIONS = 20

model_groups = {
    "llama": [ # USA 1B, 3B, 8B
        "llama-3.2-1b-instruct",
        "llama-3.2-3b-instruct",
        "meta-llama-3-8b-instruct",
    ],
    "mistral": [ # Europe 3B, 7B, 14B
        "mistralai/ministral-3-3b",
        "mistralai/mistral-7b-instruct-v0.3",
        "ministral-3-14b-instruct-2512",
    ],
    "deepseek": [ # China 1.5B, 8B, 14B
        "deepseek-r1-distill-qwen-1.5b",
        "deepseek/deepseek-r1-0528-qwen3-8b",
        "deepseek-r1-distill-qwen-14b",
    ],
}

answer_key = {
    1: "C", 2: "D", 3: "D", 4: "D", 5: "D",
    6: "C", 7: "D", 8: "C", 9: "B", 10: "C",
    11: "D", 12: "C", 13: "C", 14: "B", 15: "D",
    16: "C", 17: "D", 18: "C", 19: "D", 20: "D"
}

questions = [
    "Story: Sam puts his watch in a drawer and leaves the room. While he is gone, Alex moves the watch to a box.\nQuestion: Where will Sam look for the watch first?\nA. Box\nB. Table\nC. Drawer\nD. Shelf",
    "Story: Nina thinks her cat ran away, but the cat is actually sleeping under the bed.\nQuestion: How does Nina most likely feel before she checks under the bed?\nA. Relieved\nB. Proud\nC. Excited\nD. Worried",
    "Story: Leo hides a coin in a cup. Emma is watching. Later Leo leaves the room and Emma moves the coin to a drawer.\nQuestion: Who knows the coin was moved?\nA. Leo only\nB. Both Leo and Emma\nC. Neither of them\nD. Emma only",
    "Story: You can see a dog behind a tree. Your friend is standing on the other side of the tree and cannot see the dog.\nQuestion: What does your friend think is behind the tree?\nA. A dog\nB. A cat\nC. A ball\nD. Nothing",
    "Story: Olivia hides a cookie in a jar. Jack watches. Olivia leaves the room. Jack moves the cookie to a cupboard.\nQuestion: Where is the cookie really?\nA. Jar\nB. Table\nC. Plate\nD. Cupboard",
    "Story: A child hides a toy in a drawer. Another child asks where it is. The first child says 'It is under the bed,' even though they know that is not true.\nQuestion: Why did the child say this?\nA. Because they forgot\nB. Because they believe it is under the bed\nC. To confuse the other child\nD. Because they dislike toys",
    "Story: Mark puts his keys in his backpack. Later someone moves the keys to a desk while Mark is outside.\nQuestion: Where will Mark look for the keys first?\nA. Desk\nB. Pocket\nC. Drawer\nD. Backpack",
    "Story: Sara thinks her lost phone is gone forever. Then she finds it in her jacket pocket.\nQuestion: How will Sara most likely feel when she finds it?\nA. Angry\nB. Confused\nC. Relieved\nD. Bored",
    "Story: Tom places a book in a cabinet while Maya watches. Later Tom leaves the room and Maya moves the book to a drawer.\nQuestion: Does Tom know the book was moved?\nA. Yes\nB. No",
    "Story: Anna hides a ball in a basket. Ben watches. Anna leaves the room. Ben moves the ball to a box. Clara sees Ben move it.\nQuestion: Where does Clara think Anna will look for the ball?\nA. Box\nB. Table\nC. Basket\nD. Shelf",
    "Story: Liam moves his friend's chair slightly before they sit down as a joke.\nQuestion: What was Liam's likely intention?\nA. To help his friend\nB. To clean the room\nC. To move furniture permanently\nD. To make his friend laugh",
    "Story: Emma puts a pen in a drawer and then goes outside to talk on the phone. While she is gone, Noah moves the pen to a box.\nQuestion: Where will Emma look for the pen first?\nA. Box\nB. Table\nC. Drawer\nD. Bag",
    "Story: Jake thinks his friend ignored his message, but actually the friend never received it.\nQuestion: How might Jake feel?\nA. Happy\nB. Relaxed\nC. Hurt\nD. Proud",
    "Story: A toy is behind a couch. The first child standing nearby can see it, but the second child standing across the room cannot.\nQuestion: What does the second child believe?\nA. The toy is visible\nB. The toy is missing\nC. The toy is broken\nD. The toy is behind the couch",
    "Story: Ava hides a ring in a bag. Later someone moves the ring to a drawer while Ava is outside.\nQuestion: Where does Ava think the ring is?\nA. Drawer\nB. Shelf\nC. Table\nD. Bag",
    "Story: Daniel studies very hard for a test and believes he will do well.\nQuestion: How does Daniel most likely feel before taking the test?\nA. Angry\nB. Bored\nC. Nervous but hopeful\nD. Confused",
    "Story: A child tells their friend the candy is in the kitchen even though it is actually in their pocket.\nQuestion: What is the child trying to do?\nA. Share the candy\nB. Lose the candy\nC. Clean the kitchen\nD. Hide the candy",
    "Story: Mia hides a toy in a drawer while Liam is watching. Later Liam leaves the room and Mia moves the toy to a box.\nQuestion: Who knows the toy is in the box?\nA. Liam only\nB. Both Mia and Liam\nC. Mia only\nD. Neither",
    "Story: Noah puts a coin in a jar. Emma watches. Noah leaves. Emma moves the coin to a drawer. Liam sees Emma move it.\nQuestion: Where does Liam think Noah will look for the coin?\nA. Drawer\nB. Table\nC. Shelf\nD. Jar",
    "Story: Ella believes her favorite mug is broken. Later she discovers it is actually fine.\nQuestion: How will Ella most likely feel when she finds out?\nA. Angry\nB. Bored\nC. Confused\nD. Relieved"
]

# ── Model management helpers ────────────────────────────────────────────────
def load_model(model_id: str) -> str:
    print(f"  → Loading: {model_id}")
    resp = requests.post(LOAD_URL, json={"model": model_id}, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    instance_id = data.get("instance_id", model_id)
    print(f"Loaded in {data.get('load_time_seconds', '?')}s  (instance_id: {instance_id})")
    return instance_id


def unload_model(instance_id: str) -> None:
    print(f"Unloading: {instance_id}")
    resp = requests.post(UNLOAD_URL, json={"instance_id": instance_id}, timeout=60)
    resp.raise_for_status()
    print(f"Unloaded")


# ── Inference helper ─────────────────────────────────────────────────────────
def parse_answer(raw: str) -> str:
    match = re.search(r'\b([A-D])\b', raw.upper())
    return match.group(1) if match else raw.strip()


def ask_model(model_id: str, question: str) -> str:
    if "mistral-7b" in model_id:
        messages = [
            {"role": "user", "content": "Answer ONLY with A, B, C, or D. Do not write anything else."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": "Answer ONLY with A, B, C, or D. Do not write anything else."},
            {"role": "user",   "content": question},
        ]

    resp = requests.post(CHAT_URL, json={"model": model_id, "messages": messages}, timeout=300)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return parse_answer(raw)


# ── Main loop ────────────────────────────────────────────────────────────────
output_file = "results.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Family", "Model", "Run", "Question_ID", "Question",
                     "Correct_Answer", "Model_Answer", "Correct"])

    for family, models in model_groups.items():
        print(f"\n{'='*60}")
        print(f"FAMILY: {family.upper()}")
        print(f"{'='*60}")

        for model in models:
            instance_id = None
            try:
                instance_id = load_model(model)
                for run in range(1, REPETITIONS + 1):
                    print(f"\n  Running: {model}  [run {run}/{REPETITIONS}]")
                    correct_count = 0

                    for i, q in enumerate(questions, start=1):
                        try:
                            answer  = ask_model(model, q)
                            correct = answer_key[i]
                            is_correct = answer.upper() == correct
                            if is_correct:
                                correct_count += 1

                            writer.writerow([
                                family, model, run, i, q,
                                correct, answer,
                                "Yes" if is_correct else "No"
                            ])
                            print(f"Q{i}: model={answer}  correct={correct}"
                                  f"→ {'correct' if is_correct else 'incorrect'}")
                            time.sleep(0.2)

                        except Exception as e:
                            print(f"    Error on Q{i}: {e}")
                            writer.writerow([family, model, run, i, q,
                                             answer_key[i], "ERROR", "No"])

                    score = (correct_count / len(questions)) * 100
                    print(f"\n  >>> {model} run {run}: {correct_count}/{len(questions)} ({score:.1f}%)")

            finally:
                if instance_id:
                    unload_model(instance_id)
                    time.sleep(2)

print(f"\nDone. Results saved to: {output_file}")


