## DQN learning setup
Run Netris server at host
```bash
./netris -w -u -i 2
```

Run proxy robot at host
```bash
./netris -n -m -c localhost -i 2 -r './dqn_robot.py -t /dev/pts/3'
```

Create docker container
```bash
docker run -v $PWD:/tmp -w /tmp --gpus all -it --name tf_netris --network host tensorflow/tensorflow:latest-gpu-py3
```

Setup DQN agent
```bash
docker exec -it tf_netris python dqn.py
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