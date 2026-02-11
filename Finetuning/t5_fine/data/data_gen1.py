import json
import random

# Example shell/Oracle commands pool
commands = [
    "ps -ef | grep -i abc | grep -v grep",
    "ls -l /var/log",
    "df -h",
    "cat /etc/passwd",
    "lsnrctl status <LISTENER_NAME>",
    "lsnrctl start <LISTENER_NAME>",
    "lsnrctl stop <LISTENER_NAME>",
    "srvctl start listener -l <LISTENER_NAME>",
    "srvctl stop listener -l <LISTENER_NAME>",
    "echo 'Listener started successfully'",
    "sqlplus / as sysdba",
    "export ORACLE_HOME=/u01/app/oracle/product/19.0.0/dbhome_1",
    "tail -f alert.log",
    "grep ERROR listener.log",
    "systemctl restart oracle-listener"
]

# Example instruction fragments
sentences = [
    "For Oracle standalone [Non-RAC] database server",
    "Login to the owner account of the listener process",
    "Run the following command to check status",
    "Ensure you follow the instructions carefully",
    "Verify listener configuration parameters",
    "If the listener is running from grid home",
    "During planned maintenance",
    "After restart, confirm service availability",
    "Please make sure to document the changes",
    "Contact DBA team if the issue persists"
]

dataset = []

for _ in range(10000):
    # Each line will have between 1 and 20 commands
    num_scripts = random.randint(1, 20)
    chosen_cmds = random.choices(commands, k=num_scripts)

    parts = []
    for cmd in chosen_cmds:
        # Randomly shuffle: sometimes sentence before, sometimes after
        if random.random() > 0.5:
            parts.append(random.choice(sentences) + ": " + cmd)
        else:
            parts.append(cmd + " " + random.choice(sentences))

    # Messy single line with multiple scripts and sentences
    input_text = " ".join(parts)

    # Output = just the commands separated by newlines
    output_text = "\n".join(chosen_cmds)

    dataset.append({"input": input_text, "output": output_text})

# Save dataset
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ Generated dataset.json with {len(dataset)} entries")
