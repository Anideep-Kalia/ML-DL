import json
import random

# Paths
DATASET_PATH = "4.json"
OUTPUT_PATH = "4.json"   # overwrite with new data

# Example command pool
commands = [
    "ps -ef | grep -i abc | grep -v grep",
    "lsnrctl status <LISTENER_NAME>",
    "lsnrctl status <LISTENER_NAME>| grep \"Listener Log File\"",
    "srvctl start listener -n node_name -l listener_name",
    "srvctl stop listener -l listener_name",
    "sqlplus / as sysdba",
    "cd /var/log",
    "ls -lhS",
    "sudo gzip -9 <filename>",
    "tail -f alert.log",
    "grep ERROR listener.log",
    "systemctl restart oracle-listener"
]

# Example SOP sentences
sentences = [
    "For Oracle standalone [Non-RAC] database server",
    "Login to the owner account of the listener process",
    "Run the following command to check status",
    "Ensure you follow the instructions carefully",
    "Verify listener configuration parameters",
    "After restart, confirm service availability",
    "Please make sure to document the changes",
    "Contact DBA team if the issue persists",
    "To review the log",
    "During planned maintenance"
]

def generate_multi_command_case():
    """Generate a single multi-command SOP example."""
    num_cmds = random.randint(2, 5)  # 2–5 commands per case
    chosen_cmds = random.sample(commands, num_cmds)

    parts = []
    for cmd in chosen_cmds:
        # randomly put sentence before or after command
        if random.random() > 0.5:
            parts.append(random.choice(sentences) + ": " + cmd)
        else:
            parts.append(cmd + " " + random.choice(sentences))

    input_text = " ".join(parts)
    output_text = "\n".join(chosen_cmds)

    return {"input": input_text, "output": output_text}

def main():
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    print(f"[INFO] Current dataset size: {len(dataset)}")

    # Add 5500 multi-command cases
    for _ in range(5500):
        dataset.append(generate_multi_command_case())

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"[SUCCESS] New dataset size: {len(dataset)} (added 5500 multi-command cases)")

if __name__ == "__main__":
    main()
