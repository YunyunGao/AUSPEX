# Use the Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

SHELL ["/bin/bash","-l", "-c"]

# Copy your Conda environment file
COPY auspex_python/environment.yml /app/

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment and ensure it is activated in subsequent steps
# Note: Use ENV to set the PATH so the environment remains active in the container
ENV PATH=/opt/conda/envs/auspex/bin:$PATH

# Copy your application code
COPY auspex_python /app/auspex_python

RUN bash /opt/conda/etc/profile.d/conda.sh \
 && conda activate auspex \
 && pip install auspex_python/. \
 && echo "conda activate auspex" >> /root/.bashrc

# Set the command to run your application
ENTRYPOINT ["auspex"]
