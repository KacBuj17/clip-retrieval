FROM nvidia/cuda:13.0.0-base-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv sudo \
        make build-essential git curl \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . .

RUN chmod +x install_fixed.sh \
    && ./install_fixed.sh \
    && pip cache purge \
    && rm -rf /tmp/* /var/tmp/*

RUN rm -f install_fixed.sh setup.py Makefile requirements.txt README.md

CMD ["bash"]
