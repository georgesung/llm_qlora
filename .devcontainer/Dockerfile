FROM pytorch/pytorch
ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=UTC
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/libcudart.so
RUN apt update && apt install git net-tools curl -y
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | /bin/bash
RUN apt install git-lfs -y
RUN pip install transformers bitsandbytes datasets loralib sentencepiece scipy accelerate jupyter ipykernel fairscale && \
    pip install git+https://github.com/huggingface/peft.git
RUN git clone https://github.com/georgesung/llm_qlora.git && cd llm_qlora && pip install -r requirements.txt
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp
