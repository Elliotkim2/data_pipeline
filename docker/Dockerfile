FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum update -y && \
    yum install -y \
    git \
    libgomp \
    tar \
    xz \
    gcc \
    gcc-c++ \
    make \
    mesa-libGL \
    && yum clean all

# Download and install static ffmpeg binary
RUN mkdir -p /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz && \
    tar -xf ffmpeg.tar.xz && \
    cp ffmpeg-*/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    cd / && \
    rm -rf /tmp/ffmpeg

# Add to PATH
ENV PATH="/usr/local/bin:${PATH}"

# Environment variables for cache
ENV MPLCONFIGDIR=/tmp/mplconfig
ENV TORCH_HOME=/tmp
ENV HF_HOME=/tmp
ENV TRANSFORMERS_CACHE=/tmp
ENV XDG_CACHE_HOME=/tmp

# Set pip timeout and retries for large packages
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5 \
    PIP_RETRY_DELAY=10

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY audio_emotion_extractor_whisper.py ${LAMBDA_TASK_ROOT}
COPY pose_extractor.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD ["lambda_function.lambda_handler"]