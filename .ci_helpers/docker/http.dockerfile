FROM httpd:2.4
ARG TARGETPLATFORM

RUN apt update && apt-get update
RUN apt install -y git

COPY echopype/test_data /usr/local/apache2/htdocs/data
