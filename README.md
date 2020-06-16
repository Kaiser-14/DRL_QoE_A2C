# 5G Multimedia QoE Optimization based on Deep Reinforcement Learning algorithm

5G Telecommunications networks have transformed the current industry landscape at the level of service and application possibilities. Its improvements over the previous generation create previously inefficient use cases at production levels, such as audiovisual broadcasting of live content.

One of the most powerful and studied fields in recent years are Deep Learning and Reinforcement Learning. In general, the first is responsible for simulating neural networks to achieve greater efficiency in training Machine Learning models, while the second seeks to predict what actions an agent should take, maximizing the reward received. The combination of both areas causes an algorithm called Advantage Actor Critic (A2C).

The A2C algorithm is developed during this project, being trained through a live television signal. The system will offer the current bitrate parameters to the model, and through iterative training, the model will learn to configure the optimal settings for the state in which both the transmission and the network are based on the rewards received. This method combines the fields of Deep Learning and Reinforcement Learning since it uses neural networks for creation for both the Actor and the Critic; and second, because the algorithm itself updates its policy parameters by obtaining random actions, reinforcing those that return a greater reward.

## Getting Started

Following these instructions you will prepare the environment to test the project for development and running for testing purposes. It has been tested only in Ubuntu, but the software used is available in every OS, so take a look in each step if you are using a different OS.

### Installing

The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

Python // Docker // VirtualBox

1. Install Python [Choose a release for Linux from https://www.python.org/downloads/source/].
```bash
wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tar.xz
tar -xvf Python-3.6.9.tar
cd Python-3.6.9
./configure
make
make install
python3.6 -V -> Python3.69rc1 [/usr/bin/local/python3.6]
```

* Install pip if not installed in your machine.
```bash
sudo apt install pip3
```

* Install Python virtual environment
```bash
pip3 install virtualenv virtualenvwrapper
```

* Some necessary packages.

2. Install Java
```bash
sudo apt install default-jdk
```

3. Install Docker
```bash
sudo apt install default-jdk
```

4. Install VirtualBox
```bash
sudo apt install default-jdk
```

5. Install FFmpeg
```bash
sudo apt install ffmpeg
```

* Validate installation
```bash
ffmpeg -version
```

6. Install npm
```bash
npm install
```

### Setup

A step by step to get the development environment installed. Note that there are several parts included and need to be perfectly prepared to have the complete environment for training.

#### Apache Kafka

A distributed platform to exchange messages in terms of publisher/consumer.

1. Download Apache Kafka
```bash
wget http://www-us.apache.org/dist/kafka/2.4.0/kafka_2.13-2.4.0.tgz
tar xzf kafka_2.13-2.4.0.tgz
mv kafka_2.13-2.4.0 /usr/local/kafka
```

2. Setup Kafka Systemd Unit files
* Create systemd unit file for Zookeeper
```bash
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
```bash
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
```bash
systemctl daemon-reload
```

3. Start Zookeeper and Kafka server
```bash
sudo systemctl start zookeeper
sudo systemctl start kafka
sudo systemctl status zookeeper
sudo systemctl status kafka
```

4. Create topics
```bash
cd /usr/local/kafka
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tfm.probe.in
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic tfm.probe.out
bin/kafka-topics.sh --list --zookeeper localhost:2181
```

5. If you want to test Kafka server working, send some messages to any topic (also try from a different machine and replace localhost in the producer to the IP of the Kafka Server, to test server reachable from outside server localhost)
```bash
bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic notifications
bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic notifications --from-beginning
bin/kafka-console-producer.sh --broker-list localhost:9092 --topic notifications
```

#### vCE (virtual Compression Engine)

The vCE is composed by two virtual compression images which start FFmpeg processes that handle specific characteristics of the streaming (resolution and bitrate).

##### vCompression Resolution

1. Download vCompression Gitlab.  Resolution branch -> resolution
```bash
git clone https://github.com/5g-media/vCompression.git (Check to resolution branch)
```

2. Inside the folder, create the docker image
```bash
sudo docker build -t vcompression-res .
```

3. Create a container based on the previous image. Note: use a free port on your machine.
```bash
sudo docker run --name vcompression-res -p 3001:3001 -d vco-res
```

4. For the next steps you will need some information from the docker containers. Enter into the container and modify some parameters.
```bash
sudo docker ps -a
sudo docker exec -it vco-res /bin/bash
sudo apt update
sudo apt install nano -y
nano index.js
```

* KAFKA_IDENTIFIER. In order to avoid problems with local machine permissions, from a terminal outside the container run the next code and find UUID:
```bash
nano /usr/sbin/dmidecode
```

* For the next steps you will need some information from the docker containers. You can run the next code to check it in your local machine.
```bash
sudo docker ps -a
```

* API_PORT. Use the port used in the creation of the vco-res container. To check it use docker container information command and find it based on the actual name of the container. Note: If you have followed this instructions, you have to set it as API_PORT = 3001.

* Modify FFmpeg input and output which correspond to the streaming that you are going to use to train the model, and the IP of the docker container of the vcompression-bitrate (see following steps), respectively.

```javascript
const FFMPEG_INPUT = 'udp://224.0.1.4:5678';
const FFMPEG_OUTPUTS = 'udp://192.168.0.55';
```

5. EXTRA. You can also install the vcompression-resolution as a system service, but this project is installed as a docker service, so skip these steps if you have installed via docker.

* Install as systemd service
```bash
nano /etc/systemd/system/vcompression.service
```

* Content. Note: change user and working directory based on your actual working directory. You will also have to modify the file .env to make everything working, e.g., API ports and FFmpeg input and output.

```
[Unit]
Description=vCompression - a compression service
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user/vcompression
ExecStart=/usr/bin/node index.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

* Reload systemd daemon and start Service
```bash
systemctl daemon-reload
systemctl enable vcompression.service
systemctl start vcompression.service
systemctl status vcompression.service
```

##### vCompression Bitrate

1. Clone repository vcompression bitrate (Master branch -> bitrate).
```bash
git clone https://github.com/5g-media/vCompression.git  (Check to master branch)
```

2. Inside the folder, create the docker image
```bash
sudo docker build -t vcompression-br .
```

3. Create a container based on the previous image. Note: use a free port on your machine.
```bash
sudo docker run --name vcompression-br -p 3000:3000 -d vco-br
```

4. For the next steps you will need some information from the docker containers. Enter into the container and modify some parameters.
```bash
sudo docker ps -a
sudo docker exec -it vco-br /bin/bash
nano index.js
```

* KAFKA_IDENTIFIER. In order to avoid problems with local machine permissions, from a terminal outside the container run the next code and find UUID:
```bash
nano /usr/sbin/dmidecode
```

* API_PORT. Use the port used in the creation of the vco-br container. To check it use docker container information command and find it based on the actual name of the container. Note: If you have followed this instructions, you have to set it as API_PORT = 3000.

* Modify FFmpeg input and output which correspond to the IP address of the FFmpeg output of the vCompression-resolution, and the IP address of the virtual machine which host the probe (see following steps), respectively.

```javascript
const FFMPEG_INPUT = 'udp://224.0.1.4:5678';
const FFMPEG_OUTPUTS = 'udp://192.168.0.55';
```

5. EXTRA. You can also install the vcompression-bitrate as a system service, following the same steps and in the case of the vcompression-resolution, but remember that this project uses docker to recreate the environment.

#### Traffic Background

The goal of the project is to simulate a real environment, so it is generated some noise with a traffic background simulating the behaviour of other consumers retrieving content from the network. You can use other tool, but this project use the vCE focused on bitrate due to the ability to modify the maximum bitrate on demand.

1. Using the same vcompression-resolution image as created before, create a new container with a free port on your machine.
```bash
sudo docker run --name traffic_bg -p 3002:3002 -d vcompression-br
```

2. In the same way as the vcompression-bitrate, modify specific parameters.

* Enter into the container.
```bash
sudo docker exec -it traffic_bg /bin/bash
```

* Modify parameters in index.js.
```bash
nano index.js
```

* ```API_PORT = 3002``` (or the port that you set)
* ```const FFMPEG_INPUT = 'udp://224.0.1.4:5678';``` (IP address of any UDP streaming or local file with enough bitrate (mainly RAW content))
* ```const FFMPEG_OUTPUTS = 'udp://192.168.0.55';``` (IP address of the virtual machine where the probe is hosted; feel free to choose a free port)


#### Traffic Manager
Modify Traffic Control to handle the link bandwidth.

1. Create a file called tunnel.sh
```bash
nano tunnel.sh
```

2. Include the following lines (Check if the network interface name enp0s3 is the same, changing if neccessary).
```bash
modprobe ifb numifbs=1
ip link set dev ifb0 up
sudo tc qdisc del dev enp0s3 ingress
sudo tc qdisc del dev ifb0 root
sudo tc qdisc del dev enp0s3 root
sudo tc qdisc add dev enp0s3 handle ffff: ingress
sudo tc filter add dev enp0s3 parent ffff: protocol ipv4 u32 match u32 0 0 action mirred egress redirect dev ifb0
sudo tc qdisc add dev enp0s3 root tbf rate 20mbit burst 32kbit latency 40ms
sudo tc qdisc add dev ifb0 root tbf rate 20mbit burst 32kbit latency 40ms
```

3. Execute the previous file to create the bandwidth limit.
```bash
sh tunnel.sh
```

4. Check with any network tool such as iperf if the network parameters have been modified.

#### Probe deployment
Deploy the probe inside the Traffic Manager (virtual machine or local host to manipulate the traffic)

1. Pull images from Docker Hub and start two daemon with two probes
<!--- sudo docker pull kaiser1414/upm_tfm:1.1.3 --->
```bash
sudo docker login
sudo docker ps -a
sudo docker pull {DOCKER REPOSITORY}
sudo docker run -d --name probe -p 3005:3005 kaiser1414/upm_tfm:1.1.3
```

2. Enter into the container.
```bash
sudo docker exec -it probe /bin/bash
```

3. Modify specific files to send messages to the corresponding topics.
```bash
nano /home/test.js
```

* KAFKA_TOPIC = 'tfm.probe.out'
* var producer = Kafka.start('localhost', '9092') (or IP address and port where the Kafka server is located)

* var message_6 = {
    "vdu": obj.Mac_addr.toString(),
    "metric": "metrics",
    "value": {
        "resolution": obj.videoSettings.height.toString(),
        "frame_rate": obj.videoSettings.frame_rate.toString(),
        "bitrate": obj.videoSettings.bitrate.toString(),
        "duration": obj.videoSettings.duration.toString(),
        "mos": obj.Predictions.mos.toString()
        },
    "unit": "double",
    "timestamp": dateUTC
  };

* Modify callback function, leaving uncommented only the process.exit(1) line.

* Kafka.send(KAFKA_TOPIC, JSON.stringify(message_6), callback);

2. Move to the working directory and install some components.
```bash
cd var/www/html/proyectos/videoqualityprobe/Release/
```

3. Create a JSON with all the dependencies for NPM.
* First create the file.
```bash
nano package.json
```

* Include the following text.
```json
{
  "name": "5gmedia-vcompression-lb",
  "version": "1.0.0",
  "description": "vCompression Engine with built-in load balancer that can be controlled via a RESTful API.",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "repository": {
    "type": "git",
    "url": "git+https://github.com/BitTubes/5gmedia-vcompression-lb.git"
  },
  "keywords": [
    "ffmpeg",
    "node.js",
    "fastify",
    "load",
    "balancer",
    "vnf",
    "5g-media"
  ],
  "author": "Igor Fritzsch",
  "license": "UNLICENSED",
  "bugs": {
    "url": "https://github.com/BitTubes/5gmedia-vcompression-lb/issues"
  },
  "homepage": "https://github.com/BitTubes/5gmedia-vcompression-lb#readme",
  "dependencies": {
    "dotenv": "^7.0.0",
    "fastify": "^1.14.1",
    "kafka-node": "^4.0.2",
    "macaddress": "^0.2.9",
    "node-media-server": "^1.4.10",
    "pidusage": "^2.0.17",
    "request": "^2.88.0"
  }
}
```

4. Install node.js.
```bash
npm install
```

5. Create the index.js which will control the behaviour of the probe.
* First create the file.
```bash
nano index.js
```

* Include the following text.
```javascript
const fs = require('fs');
const {spawn} = require('child_process');
const fastify = require('fastify')({
  logger: true
});


///////////// Vars ////////////
const API_PORT = 3005;

const FFMPEG_INPUT = 'udp://172.17.0.6:1234'

const args = [
  '-i', FFMPEG_INPUT,
  '-x', '5gmedia',
  '-n', '1000'
];

function spawnProbe(){
  probe = spawn('./videoqualityprobe', args)
        .once('close', () => {
          console.log('Probe closed. Restarting...');
          spawnProbe();
        });
}

spawnProbe();

///////////// API /////////////

fastify.get('/', (request, reply) => {
  reply.send({status: true});
});

fastify.post('/refresh/', (request, reply) => {
  reply.send({status: true});
  probe.kill();
});

fastify.listen(API_PORT, '0.0.0.0', (err, address) => {
  if (err) throw err;
  fastify.log.info(`server listening on ${address}`)
});
```

5. Need to explain adding index.js to avoid probe failing

#### Deep Reinforcement Learning model

1. Create new virtual environment
```bash

which python3 -> e.g. /usr/bin/python3
virtualenv venv --python=/usr/bin/python3

```

2. Activate the virtual environment
```bash
source venv/bin/activate
```

3. Install necessary requirements
```bash
pip install -r requirements.txt
```

4. Clone repository
```bash
git clone https://github.com/Kaiser-14/DRL_QoE_A2C.git
```
5. If you are done working with the virtual environment
```bash
deactivate
```

## Project running

Enable each part to start the whole system

1. Start FFmpeg process to transmit to the Docker 'probe_in' direction
```bash
ffmpeg -re -i "udp://224.0.1.4:5678?overrun_nonfatal=1&fifo_size=50000000" -c:v copy -c:a copy -f mpegts udp://172.17.0.2:5678
```

2. Enable vcompression focused on handling resolution.
```bash
sudo docker exec -it vco-res /bin/bash/
node index.js
```

* If the ffmpeg processes are continuosly closing, restart the container.
```bash
exit (inside the container)
sudo docker restart vco-res (in the same terminal after exit the container)
sudo docker exec -it vco-res /bin/bash/
```

* vcompression-res has some APIs to interact via terminal. Note: use the same port that you set in the creation of the container
```bash
curl -X GET http://localhost:3000/ (receive actual information of the streaming process)
curl -X POST http://localhost:3000/resolution/high (modify the resolution to 1080p)
curl -X POST http://localhost:3000/resolution/low (modify the resolution to 720p)
```

3. Enable vcompression focused on handling bitrate.
```bash
sudo docker exec -it vco-br /bin/bash/
node index.js
```

* If the ffmpeg processes are continuosly closing, restart the container.
```bash
exit (inside the container)
sudo docker restart vco-br (in the same terminal after exit the container)
sudo docker exec -it vco-br /bin/bash/
node index.js
```

* vcompression-br has some APIs to interact via terminal. Note: use the same port that you set in the creation of the container
```bash
curl -X GET http://localhost:3001/ (to receive actual information of the streaming process)
curl -X POST http://localhost:3001/bitrate/10000 (modify the maximum bitrate)
curl -X POST http://localhost:3001/refresh/ (refresh the internal FFmpeg)
```

4. Enable Traffic Background
```bash
sudo docker exec -it traffic_bg /bin/bash
```
* run the process.
```bash
sudo docker exec -it traffic-bg /bin/bash
node index.js
```

* traffic-bg has the same APIs as the vco-br except the refresh API.
```bash
curl -X GET http://localhost:3002/ (receive actual information of the streaming process)
curl -X POST http://localhost:3002/bitrate/10000 (modify the maximum bitrate)
```

5. Enable probe
```bash
sudo docker exec -it probe /bin/bash
cd var/www/html/proyectos/videoqualityprobe/Release/
node index.js
```

6. Execute the model
```bash
python3 agent.py
```
