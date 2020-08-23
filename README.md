## Docker setup
Create docker container
```bash
docker run -v $PWD:/tmp -w /tmp --gpus all -it --name tf_netris --network host tensorflow/tensorflow:latest-gpu-py3
```

## DQN learning setup
Run Netris server at host
```bash
./netris-env -w -u -i 0.1
```

Run dqn agent with CPU at host
```bash
python dqn.py
```

Run dqn agent with GPU support at guest
```bash
docker start tf_netris
docker exec -it tf_netris python dqn.py -g -p 9800
```

Run proxy robot at host (note that interval (`-i`) must match value passed to Netris server
```bash
./netris-env -n -m -c localhost -i 0.1 -r 'python env_proxy.py -t /dev/pts/3 -p 9800'
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
