import json
import random

# -------------------- Config -------------------- #
OLD_DATASET_PATH = "3.json"          # your original dataset
NEW_DATASET_PATH = "4.json"  # output dataset with appended cases
NUM_NEW_CASES = 2000

# -------------------- Problematic Cases -------------------- #
special_cases = [
    {
        "input": "cd /",
        "output": "cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /cd /",
        "correct": "cd /"
    },
    {
        "input": "If you identify /var/cache/yum consuming space from step 2:",
        "output": "var/cache/yum consuming space",
        "correct": "#If you identify /var/cache/yum consuming space from step 2:"
    },
    {
        "input": "If you identify logs in /opt/oracle.ExaWatcher/archive from step 2:",
        "output": "opt/oracle.ExaWatcher/archive",
        "correct": "# If you identify logs in /opt/oracle.ExaWatcher/archive from step 2:"
    },
    {
        "input": "Compress old logs (if not already rotated):",
        "output": "Compress old logs",
        "correct": "#Compress old logs (if not already rotated):"
    },
    {
        "input": "compress only file of names messages-*",
        "output": "compress only file of names messages-*",
        "correct": "#compress only file of names messages-*"
    }
]

# -------------------- Main Logic -------------------- #
def augment_dataset():
    # Load old dataset
    with open(OLD_DATASET_PATH, "r") as f:
        old_data = json.load(f)

    # Generate 2000 new problematic cases
    new_cases = [random.choice(special_cases) for _ in range(NUM_NEW_CASES)]

    # Append to old dataset
    augmented_data = old_data + new_cases

    # Save new dataset
    with open(NEW_DATASET_PATH, "w") as f:
        json.dump(augmented_data, f, indent=4)

    print(f"[SUCCESS] Augmented dataset saved to {NEW_DATASET_PATH} with {len(augmented_data)} total samples.")

if __name__ == "__main__":
    augment_dataset()
