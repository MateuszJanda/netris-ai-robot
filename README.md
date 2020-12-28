## Docker setup
Create docker container
```bash
docker run -v $PWD:/tmp -w /tmp --gpus all -it --name tf_netris --network host tensorflow/tensorflow:latest-gpu-py3
```

## DQN learning setup
On first terminal run Netris server (at host)
```bash
./netris-env -w -u -i 0.1
```

On second terminal, run dqn agent (with GPU support at guest)
```bash
docker start tf_netris
docker exec -it tf_netris python dqn.py -g -p 9800
```

Alternatively, you can run dqn agent with CPU support (at host)
```bash
python dqn.py
```

On third terminal, run proxy robot (at host). Note that interval (`-i`) must match value passed to Netris server
```bash
./netris-env -n -m -c localhost -i 0.1 -r 'python env_proxy.py -t /dev/pts/3 -p 9800'
```

## Example
Learned model (in supervised learning), can be used by robot with normal netris instance
On first termianl, run game in server mode and wait for **robot**
```bash
netris -w
```
On second terminal, connect **robot** to server as second player
```bash
netris -c localhost -r 'python robot_sl.py -f'
```
