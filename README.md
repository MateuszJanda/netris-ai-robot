## DQN learning setup
Run Netris server at host
```bash
./netris-env -w -u -i 0.1
```

Run proxy robot at host
```bash
./netris-env -n -m -c localhost -i 0.1 -r 'python dqn_proxy.py -t /dev/pts/3'
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
netris -c localhost -r 'python robot_sl.py -f'
```

## Setup
https://www.tensorflow.org/install
pip install tensorflow