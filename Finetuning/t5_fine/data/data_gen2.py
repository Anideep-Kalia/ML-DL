import json
import random

# Path to your original dataset
DATASET_PATH = "1.json"
OUTPUT_PATH = "3.json"

# Error scenarios with (input, correct) pairs
error_scenarios = [
    ("ps -ef | grep lsnr | grep -v grep.", "ps -ef | grep lsnr | grep -v grep"),
    ("ps -ef | grep lsnr | grep -v grep | awk '{print $9}' > listeners_${HOSTNAME}.txt", 
     "ps -ef | grep lsnr | grep -v grep | awk '{print $9}' > listeners_${HOSTNAME}.txt"),
    ("for lsnr in cat listeners_${HOSTNAME}.txt do", "for lsnr in cat listeners_${HOSTNAME}.txt do"),
    ("above grep command will provide the listener names which are running on the server", 
     "# above grep command will provide the listener names which are running on the server"),
    ("SQL>show parameter LOCAL_LISTENER;", "SQL>show parameter LOCAL_LISTENER;"),
    ("srvctl start listener -n node_name -l listener_name", "srvctl start listener -n node_name -l listener_name"),
    ("srvctl start scan_listener [-node node_name] [-scannumber ordinal_number]", 
     "srvctl start scan_listener [-node node_name] [-scannumber ordinal_number]"),
    ("cd /var/log", "cd /var/log"),
    ("ls -lhS", "ls -lhS"),
    ("sudo gzip -9 <filename>", "sudo gzip -9 <filename>")
]

def generate_variations(base_input, correct_output, n=300):
    """Generate n variations of an error case with small random tweaks."""
    examples = []
    for i in range(n):
        variation = base_input
        # Add random extra spaces, punctuation, or prefixes
        if random.random() > 0.5:
            variation = " ".join(variation.split())  # normalize spaces
        if random.random() > 0.7:
            variation = variation.replace(" ", "  ")  # double spaces
        if random.random() > 0.8:
            variation = variation + random.choice(["", ".", "   ", ":"])
        examples.append({
            "input": variation,
            "output": correct_output
        })
    return examples

def main():
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    augmented = dataset.copy()
    for inp, correct in error_scenarios:
        augmented.extend(generate_variations(inp, correct, 300))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"[SUCCESS] Augmented dataset saved to {OUTPUT_PATH}")
    print(f"Original size: {len(dataset)}, New size: {len(augmented)}")

if __name__ == "__main__":
    main()
