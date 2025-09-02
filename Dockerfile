FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip python3-venv sudo \
    make build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . .

RUN chmod +x install_fixed.sh
RUN ./install_fixed.sh

CMD ["bash"]
