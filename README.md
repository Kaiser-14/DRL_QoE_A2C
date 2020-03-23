```

```
# Apache Kafka installation
1. Install Java
```
sudo apt install default-jdk
```

2. Download Apache Kafka
```
wget http://www-us.apache.org/dist/kafka/2.4.0/kafka_2.13-2.4.0.tgz
tar xzf kafka_2.13-2.4.0.tgz
mv kafka_2.13-2.4.0 /usr/local/kafka
```

3. Setup Kafka Systemd Unit files
* Create systemd unit file for Zookeeper
```
nano /etc/systemd/system/zookeeper.service
```
* And the content
```
[Unit]
Description=Apache Zookeeper server
Documentation=http://zookeeper.apache.org
Requires=network.target remote-fs.target
After=network.target remote-fs.target

[Service]
Type=simple
ExecStart=/usr/local/kafka/bin/zookeeper-server-start.sh /usr/local/kafka/config/zookeeper.properties
ExecStop=/usr/local/kafka/bin/zookeeper-server-stop.sh
Restart=on-abnormal

[Install]
WantedBy=multi-user.target
```

* Create systemd unit file for Kafka
```
nano /etc/systemd/system/kafka.service
```

* And the content
```
[Unit]
Description=Apache Kafka Server
Documentation=http://kafka.apache.org/documentation.html
Requires=zookeeper.service

[Service]
Type=simple
Environment="JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64"
ExecStart=/usr/local/kafka/bin/kafka-server-start.sh /usr/local/kafka/config/server.properties
ExecStop=/usr/local/kafka/bin/kafka-server-stop.sh

[Install]
WantedBy=multi-user.target
```

* Reload system daemon
```
systemctl daemon-reload
```

4. Start Zookeeper and Kafka server
```
sudo systemctl start zookeeper
sudo systemctl start kafka
sudo systemctl status zookeeper
sudo systemctl status kafka
```

5. Create topics
```
cd /usr/local/kafka
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tfm.probe.in
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tfm.probe.out
bin/kafka-topics.sh --list --zookeeper localhost:2181
```

5. If you want to test Kafka server working, send some messages to any topic (and try from a different machine and replace localhost in the producer to the IP of the Kafka Server)
```
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic notifications
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic notifications --from-beginning
bin/kafka-console-producer.sh --broker-list localhost:9092 --topic notifications
```

# Multimedia handling

1. Install FFmpeg
```
sudo apt install ffmpeg
```

2. Validate installation
```
ffmpeg -version
```

3. To test software
```
ffmpeg -i input.mp4 output.webm
```

4. Start FFmpeg process to transmit to the Docker 'probe_in' direction
```
ffmpeg -re -i "udp://224.0.1.4:5678?overrun_nonfatal=1&fifo_size=50000000" -c:v copy -c:a copy -f mpegts udp://172.17.0.2:5678
ffmpeg -re -i "udp://224.0.1.4:5678?overrun_nonfatal=1&fifo_size=50000000" -c:v copy -c:a copy -f mpegts udp://172.17.0.3:5678
```

# Probe deployment

1. Pull images from Docker Hub and start two daemon with two probes
```
sudo docker login
sudo docker ps -a
sudo docker pull kaiser1414/upm_tfm:1.1.3
sudo docker run -d --name probe -p 3007:3007 kaiser1414/upm_tfm:1.1.3
```

2. Check images running and enter them in different terminals
```
sudo docker ps -a
sudo docker exec -it probe-in bash (Terminal 1)
sudo docker exec -it probe-out bash (Terminal 2)
```

3. Modify specific files to send messages to the corresponding topics (both probes). The Kafka topics must be 'tfm.probe.in' and 'tfm.probe.out' and the IP:PORT direction must correspond to the Kafka Server and Kafka Port.
```
nano /home/test.js
```

4. Enable both probes, receiving input from a ffmpeg
```
./videoqualityprobe -i udp://172.17.0.2:5678 -x 5gmedia -n 1000 (Terminal 1)
./videoqualityprobe -i udp://172.17.0.3:5678 -x 5gmedia -n 1000 (Terminal 2)
```

5. Need to explain adding index.js to avoid probe failing

# vCompression

1. Download vCompression Gitlab. Master branch -> bitrate; Resolution branch -> resolution
```
git clone
```

2. Install libraries
```
npm install
```

3. Note. KAFKA_IDENTIFIER set by hand, extracting from /usr/sbin/dmidecode. To find it:
```
nano /usr/sbin/dmidecode
```

4. Activate vCompression
```
node index.js
```

API:
GET /                   Finds vcompression's status
POST /bitrate/{value}   Set vcompression's bitrate [kbps]

Example usage
curl -X POST http://localhost:3000/bitrate/3000

5. Install as systemd service
nano /etc/systemd/system/vcompression.service

* Content
```
[Unit]
Description=vCompression - a compression service
After=network.target

[Service]
Type=simple
User=irt
WorkingDirectory=/home/irt/vcompression
ExecStart=/usr/bin/node index.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

* Reload systemd daemon and start Service
```
systemctl daemon-reload
systemctl enable vcompression.service
systemctl start vcompression.service
systemctl status vcompression.service
```


# Deep Reinforcement Learning model

1. Install Python
```
wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tar.xz [Choose a release for Linux from https://www.python.org/downloads/source/]
tar -xvf Python-3.6.9.tar
cd Python-3.6.9
./configure
make
make install
python3.6 -V -> Python3.69rc1 [/usr/bin/local/python3.6]
```

2. Install Python virtual environment
```
pip install virtualenv virtualenvwrapper
```

3. Create new virtual environment
```
which python3.6 -> e.g. /usr/bin/python3.6
virtualenv --python=/usr/bin/python3.6 ~/virtualenv/py3.6 [Linux/Mac users only]
conda create -n py3.6 python=3.6 anaconda [Windows users only]
```

4. Activate it
source ~/virtualenv/py3.6 [Mac/Linux users only]
conda activate py3.6 [Windows users only]

5. Install neccesary packages
pip3 install tflearn tensorflow matplotlib

6. Clone repository
git clone ...

7. Execute the model
python3 agent.py
