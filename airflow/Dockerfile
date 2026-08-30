FROM apache/airflow:3.3.1

USER root

# Install Java for PySpark
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

# Install project dependencies
RUN pip install --no-cache-dir \
    great-expectations \
    pyspark==4.2.0 \
    matplotlib