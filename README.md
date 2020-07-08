## DQN learning setup
Setup Netris server
```bash
./netris -w -u -i 1
```

Setup proxy robot
```bash
./netris -n -m -c localhost -i 1 -r './dqn_robot.py --log-in-terminal /dev/pts/3'
```

Setup DQN agent
```bash
python dqn.py
```

## Example
Run game and wait for **robot**
```bash
netris -w
```
Connect **robot** to server as second player
```bash
netris -c localhost -r './robot.py --log-to-file'
```

## Setup
https://www.tensorflow.org/install
pip install tensorflow